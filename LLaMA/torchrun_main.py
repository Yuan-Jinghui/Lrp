import os
import inspect
import sys
import math
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

# Set HuggingFace token from environment variable
# Note: Set HF_TOKEN environment variable before running this script

# Login to HuggingFace
try:
    from huggingface_hub import login
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("Successfully logged into HuggingFace")
    else:
        print("Warning: HF_TOKEN not found in environment variables")
        print("Using public datasets without HuggingFace authentication")
except Exception as e:
    print(f"Warning: Could not log into HuggingFace: {e}")
    print("Using public datasets without HuggingFace authentication")

# Set WANDB API key from environment variable
if 'WANDB_API_KEY' in os.environ:
    print("WANDB API key found in environment")
else:
    print("Warning: WANDB_API_KEY not found in environment")

transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser(description="Muon & RNNP Training Pipeline")

    # Model and training parameters
    parser.add_argument("--model_config", type=str, default='configs/llama_60m.json')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_matrix", type=float, default=None, help="Learning rate for matrix parameters (Muon/RMNP/Shampoo/SOAP)")
    parser.add_argument("--lr_adam", type=float, default=None, help="Learning rate for Adam parameters (Muon/RMNP/Shampoo/SOAP)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_batch_size", type=int, default=512)
    parser.add_argument("--num_training_steps", type=int, default=20000,
                        help="Number of update steps to train for.")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, default='muon_rmnp_training')
    parser.add_argument("--target_eval_tokens", type=int, default=10_000_000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--continue_from", type=str, default=None) 
    
    # Optimizer selection: muon, RMNP, shampoo, soap
    parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "RMNP", "shampoo", "soap", "new_optimizer"])
    parser.add_argument("--precondition_frequency", type=int, default=10, help="Preconditioner update frequency (SOAP only)")

    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps.")

    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--single_cuda", default=False, action="store_true")
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--r", type=float, default=1.833, help="The r value for L_{r, \\infty} norm in New_Optimizer (default: 1.833)")
    parser.add_argument("--local_data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "c4_local"), help="Path to pre-downloaded local dataset")
    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, target_eval_tokens, local_data_dir=None):
    _time = time.time()
    
    if local_data_dir is not None:
        val_data = datasets.load_from_disk(os.path.join(local_data_dir, "validation"))
        val_data = val_data.to_iterable_dataset()
        val_data = val_data.shuffle(seed=42, buffer_size=10000)
    else:
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
        val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if world_size > 1:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()} 
        labels = batch["input_ids"].clone() 
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss 
        total_loss += loss.detach() 

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size 

    total_loss = total_loss / total_batches 

    # Only gather if distributed
    if world_size > 1:
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)] 
        dist.all_gather(gathered_losses, total_loss) 
        total_loss = sum([t.item() for t in gathered_losses]) / world_size
    else:
        total_loss = total_loss.item()
    return total_loss, evaluated_on_tokens 


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not "LOCAL_RANK" in os.environ:
        os.environ['RANK'] = '0'
        os.environ["LOCAL_RANK"] = '0'
        os.environ["WORLD_SIZE"] = '1'
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = '26000'

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    # Always initialize process group for muon compatibility
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
        logger.info("Process group initialized for multi-GPU")
    else:
        # Initialize single-process group for muon optimizer compatibility
        dist.init_process_group(backend="gloo", rank=global_rank, world_size=world_size)
        logger.info("Single process group initialized for muon compatibility")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"
    
    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb_project = os.environ.get('WANDB_PROJECT', 'llama-pretraining')
        wandb.init(project=wandb_project, name=args.wandb_name)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # Load training dataset
    logger.info("Loading training dataset...")

    seed_for_shuffle = 42
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")

    if args.local_data_dir is not None:
        logger.info(f"Loading local dataset from {args.local_data_dir}/train")
        data = datasets.load_from_disk(os.path.join(args.local_data_dir, "train"))
        data = data.to_iterable_dataset()
        data = data.shuffle(seed=seed_for_shuffle, buffer_size=10000)
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    if world_size > 1:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from,'pytorch_model.bin')
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}")
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    
    # Initialize wandb config
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Setup optimizer - only muon and RMNP supported
    if args.optimizer.lower() == "muon":
        from optimizers.muon_optimizer import get_muon_optimizer
        lr_muon = args.lr_matrix if args.lr_matrix is not None else args.lr
        lr_adamw = args.lr_adam if args.lr_adam is not None else 0.001
        optimizer = get_muon_optimizer(model, lr_muon=lr_muon, lr_adamw=lr_adamw, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "rmnp":
        from optimizers.RMNP_optimizer import get_rmnp_optimizer
        # Use separate learning rates - both must be specified for RMNP
        if args.lr_matrix is None or args.lr_adam is None:
            raise ValueError("RMNP requires both --lr_matrix and --lr_adam to be specified")
        optimizer = get_rmnp_optimizer(
            model,
            lr_rmnp=args.lr_matrix,
            lr_adam=args.lr_adam,
            momentum=0.95,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "shampoo":
        from optimizers.shampoo_optimizer import get_shampoo_optimizer
        if args.lr_matrix is None or args.lr_adam is None:
            raise ValueError("shampoo requires both --lr_matrix and --lr_adam to be specified")
        optimizer = get_shampoo_optimizer(
            model,
            lr_shampoo=args.lr_matrix,
            lr_adam=args.lr_adam,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "soap":
        from optimizers.soap_optimizer import get_soap_optimizer
        if args.lr_matrix is None or args.lr_adam is None:
            raise ValueError("soap requires both --lr_matrix and --lr_adam to be specified")
        optimizer = get_soap_optimizer(
            model,
            lr_soap=args.lr_matrix,
            lr_adam=args.lr_adam,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency
        )
    elif args.optimizer.lower() == "new_optimizer":
        from optimizers.new_optimizer import get_new_optimizer
        if args.lr_matrix is None or args.lr_adam is None:
            raise ValueError("new_optimizer requires both --lr_matrix and --lr_adam to be specified")
        optimizer = get_new_optimizer(
            model,
            lr_rmnp=args.lr_matrix,  
            lr_adam=args.lr_adam,    
            r=args.r,                
            momentum=0.95,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported. Choose from: muon, RMNP, shampoo, soap")
    
    print('*********************************')
    print(f"Using optimizer: {args.optimizer}")
    print(optimizer)
    print('*********************************')

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio)

    # Load optimizer and scheduler state if continuing from checkpoint
    if args.continue_from is not None:
        optimizer_path = os.path.join(args.continue_from, "optimizer.pt")
        
        if os.path.exists(optimizer_path):
            logger.info(f"Loading optimizer state from {optimizer_path}")
            checkpoint = torch.load(optimizer_path, map_location=device)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                optimizer.load_state_dict(checkpoint)
                
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logger.warning(f"Optimizer state not found at {optimizer_path}")

    if world_size > 1:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False)

    # Training loop
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0
    
    step_times = []
    consecutive_data_errors = 0
    max_data_errors = 25  # Maximum consecutive data loading errors before giving up
    data_errors = [
        'connection', 'timeout', 'network', 'http', 'ssl',
        'socket', 'dns', 'resolve', 'unreachable', 'refused',
        'dataset', 'download', 'streaming', 'hf://', 'gzip://',
        'file not found', 'filenotfounderror'
    ]

    data_iterator = iter(dataloader)
    while update_step < args.num_training_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            logger.info("Dataloader exhausted, ending training loop.")
            break
        except Exception as e:
            consecutive_data_errors += 1
            error_msg = str(e).lower()
            is_data_error = any(error_keyword in error_msg for error_keyword in data_errors)
            if is_data_error and consecutive_data_errors <= max_data_errors:
                logger.warning(f"Data iterator error (attempt {consecutive_data_errors}/{max_data_errors}): {e}")
                logger.info("Skipping this batch and continuing...")
                time.sleep(min(consecutive_data_errors * 1.0, 10.0))
                continue
            logger.error(f"Critical data iterator error or too many retries: {e}")
            raise

        try:
            global_step += 1
            local_step += 1
            consecutive_data_errors = 0
            
            # Time first 10 steps
            if global_step <= 10:
                start_time = time.time()
                
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
            loss = model(**batch, labels=labels).loss
            loss_temp = loss.detach().clone()
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            
        except Exception as e:
            consecutive_data_errors += 1
            error_msg = str(e).lower()
            is_data_error = any(error_keyword in error_msg for error_keyword in data_errors)
            
            if is_data_error and consecutive_data_errors <= max_data_errors:
                logger.warning(f"Data loading error (attempt {consecutive_data_errors}/{max_data_errors}): {e}")
                logger.info("Skipping this batch and continuing...")
                
                global_step -= 1
                local_step -= 1
                
                time.sleep(min(consecutive_data_errors * 1.0, 10.0))
                continue
            else:
                logger.error(f"Critical error or too many consecutive data errors: {e}")
                raise e
                
        if global_step % args.gradient_accumulation != 0:
            continue
            
        # Gradient clipping
        if args.grad_clipping != 0.0:
            model_params = [p for p in model.parameters() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(model_params, args.grad_clipping)
            
        if global_rank == 0:
            pbar.update(1)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_step += 1
        
        # Time first 10 steps
        if global_step <= 10:
            step_time = time.time() - start_time
            step_times.append(step_time)
            logger.info(f"Step {global_step} time: {step_time:.2f} seconds")
            
        # Save checkpoint
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            if getattr(model_to_save, 'generation_config', None) is not None:
                model_to_save.generation_config.pad_token_id = 0
            model_to_save.save_pretrained(current_model_directory, max_shard_size='100GB', safe_serialization=False)
            
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")
            
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
            
        # Evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, target_eval_tokens=args.target_eval_tokens, local_data_dir=args.local_data_dir
            )
            if global_rank == 0:
                wandb.log({
                    "eval_loss": total_loss,
                    "eval_tokens": evaluated_on_tokens,
                    "perplexity": math.exp(total_loss),
                    },
                    step=update_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
            
        # Logging
        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size
        
        if global_rank == 0:
            log_dict = {
                "loss": loss_temp.item(),
                "lr": lr,
                "weight_decay": args.weight_decay,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
            }
            wandb.log(log_dict, step=update_step)
            
        update_time = time.time()
        
        # Early termination
        if update_step >= args.num_training_steps:
            break
    
    # Training finished
    if global_rank == 0 and len(step_times) > 0:
        avg_time = np.mean(step_times)
        est_total = avg_time * args.num_training_steps
        logger.info(f"Average step time (first 10): {avg_time:.2f} seconds")
        logger.info(f"Estimated total training time: {est_total/3600:.2f} hours for {args.num_training_steps} steps")

    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    # Final save
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0:
        logger.info(f"Saving final model to {current_model_directory}")
        os.makedirs(args.save_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        if getattr(model_to_save, 'generation_config', None) is not None:
            model_to_save.generation_config.pad_token_id = 0
        model_to_save.save_pretrained(current_model_directory, safe_serialization=False)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, loss_temp, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, target_eval_tokens=args.target_eval_tokens, local_data_dir=args.local_data_dir
    )

    if global_rank == 0:
        wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                "final_perplexity": math.exp(total_loss),
            },
            step=update_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting Muon & RNNP Training Pipeline")
    args = parse_args(None)
    main(args)

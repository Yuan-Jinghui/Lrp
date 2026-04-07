import datasets
import argparse
import os


def stream_split(split_name, limit):
    data = datasets.load_dataset("allenai/c4", "en", split=split_name, streaming=True)
    for i, item in enumerate(data):
        if i >= limit:
            break
        yield item
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_examples", type=int, default=5200000)
    parser.add_argument("--val_examples", type=int, default=40000)
    parser.add_argument("--save_path", type=str, default="./c4_local")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print(f"Downloading {args.train_examples} train examples...")
    train_local = datasets.Dataset.from_generator(
        stream_split,
        gen_kwargs={"split_name": "train", "limit": args.train_examples},
    )
    train_local.save_to_disk(os.path.join(args.save_path, "train"))

    print(f"Downloading {args.val_examples} validation examples...")
    val_local = datasets.Dataset.from_generator(
        stream_split,
        gen_kwargs={"split_name": "validation", "limit": args.val_examples},
    )
    val_local.save_to_disk(os.path.join(args.save_path, "validation"))
    print(f"Dataset successfully saved to {args.save_path}")

if __name__ == "__main__":
    main()

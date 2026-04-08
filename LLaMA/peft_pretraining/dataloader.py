from torch.utils.data import IterableDataset


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        # Upstream dataset partitioning (rank/world_size + DataLoader internals) already
        # handles worker/rank sharding. Avoid a second manual split here.
        iter_data = iter(self.data)

        text_batch = []
        for example in iter_data:
            text_batch.append(example["text"])

            if len(text_batch) == self.batch_size:
                yield self._tokenize_batch(text_batch)
                text_batch = []

        if text_batch:
            yield self._tokenize_batch(text_batch)

    def _tokenize_batch(self, texts):
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

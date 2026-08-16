"""Prompt-masked SFT dataset: the model is only trained to predict the
completion (move sequence) tokens; prompt tokens are masked out of the loss
with label = -100, the standard supervised-fine-tuning trick."""
import json
from typing import List

import torch
from torch.utils.data import Dataset


class PathfindingSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = self.tokenizer.encode(ex["prompt"])
        completion_ids = self.tokenizer.encode(ex["completion"])
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            completion_ids = completion_ids + [eos_id]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)

        if len(input_ids) > self.max_length:
            # Truncate from the left of the prompt so the completion (what we
            # actually train on) is always kept intact.
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = x["input_ids"]
            labels[i, :n] = x["labels"]
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    return collate

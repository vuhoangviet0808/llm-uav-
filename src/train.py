"""LoRA fine-tuning entry point.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/smoke_test.yaml   # tiny offline sanity check
"""
import argparse
import os

import yaml
from transformers import Trainer, TrainingArguments

from src.model import apply_lora, load_base_model
from src.sft_dataset import PathfindingSFTDataset, make_collate_fn
from src.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]

    tokenizer = get_tokenizer(model_name)
    tiny_vocab_size = tokenizer.vocab_size if model_name == "tiny-debug" else None

    print(f"Loading base model '{model_name}' ...")
    model = load_base_model(model_name, tiny_vocab_size=tiny_vocab_size)
    if model_name == "tiny-debug":
        # The tiny debug model is randomly initialized (no pretraining), so
        # there is no pretrained knowledge for LoRA to adapt -- LoRA would
        # freeze the (also random) embeddings/output head and barely move
        # the loss. Full fine-tune it directly instead; this still exercises
        # the exact same Trainer/dataset/eval code path as the real run.
        print("tiny-debug model: skipping LoRA, doing a full fine-tune instead.")
    else:
        model = apply_lora(
            model,
            r=cfg["lora"]["r"],
            alpha=cfg["lora"]["alpha"],
            dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"].get("target_modules"),
        )

    data_dir = cfg["data"]["out_dir"]
    train_ds = PathfindingSFTDataset(os.path.join(data_dir, "train.jsonl"), tokenizer, max_length)
    val_ds = PathfindingSFTDataset(os.path.join(data_dir, "val.jsonl"), tokenizer, max_length)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    tcfg = cfg["train"]
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        bf16=tcfg.get("bf16", False),
        fp16=tcfg.get("fp16", False),
        report_to=[],
        max_steps=tcfg.get("max_steps", -1),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    os.makedirs(tcfg["output_dir"], exist_ok=True)
    model.save_pretrained(tcfg["output_dir"])
    print(f"Saved LoRA adapter to {tcfg['output_dir']}")


if __name__ == "__main__":
    main()

"""LoRA fine-tuning entry point.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/smoke_test.yaml   # tiny offline sanity check
"""
import argparse
import os
import time

import torch
import yaml
from transformers import Trainer, TrainingArguments, TrainerCallback

from src.model import apply_lora, load_base_model
from src.sft_dataset import PathfindingSFTDataset, make_collate_fn
from src.tokenizer import get_tokenizer


class StepTimerCallback(TrainerCallback):
    """Prints wall-clock time for each of the first few optimizer steps.

    `logging_steps` only controls when *loss* gets printed -- it says nothing
    about whether the run is alive. On a run that silently fell back to CPU
    (e.g. Kaggle/Colab GPU quota exhausted, or the accelerator setting wasn't
    actually applied), the tqdm bar can sit at "0/N" for hours because even
    a single step genuinely takes that long for a 1B+ parameter model on CPU
    -- this makes that visible immediately instead of only after
    `logging_steps` steps complete (which may never happen within a
    reasonable wait).
    """

    def __init__(self, num_steps_to_time=5):
        self.num_steps_to_time = num_steps_to_time
        self._t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= self.num_steps_to_time and self._t0 is not None:
            elapsed = time.time() - self._t0
            print(f"[step timer] optimizer step {state.global_step} took {elapsed:.1f}s", flush=True)
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    # Print device status FIRST, before anything slow (tokenizer/model
    # download, dataset tokenization) -- so if the GPU didn't attach (quota
    # exhausted, accelerator setting not applied, etc.) that's obvious right
    # at the top of the log instead of discovered after waiting on a stuck
    # progress bar.
    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_ok}", flush=True)
    if cuda_ok:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print(
            "WARNING: no GPU detected by torch -- training will run on CPU. "
            "For a 1B+ parameter model, a single optimizer step on CPU can "
            "take many minutes to hours, which looks identical to a hang on "
            "the progress bar. If you expected a GPU here, stop this run now "
            "and check the notebook's Accelerator setting and GPU quota "
            "before re-running.",
            flush=True,
        )

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
        callbacks=[StepTimerCallback()],
    )
    print(f"Model device: {next(model.parameters()).device}", flush=True)
    print(f"Starting training: {len(train_ds)} train examples, "
          f"{training_args.max_steps if training_args.max_steps > 0 else 'auto'} max_steps "
          f"(auto = derived from num_train_epochs)", flush=True)
    trainer.train()

    os.makedirs(tcfg["output_dir"], exist_ok=True)
    model.save_pretrained(tcfg["output_dir"])
    print(f"Saved LoRA adapter to {tcfg['output_dir']}")


if __name__ == "__main__":
    main()

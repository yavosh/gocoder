#!/usr/bin/env python3
"""Fine-tune Nemotron-Cascade-2 with LoRA for Go code generation.

Usage:
    python training/train.py --config training/config/nemotron-cascade-2.yaml
"""
import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict) -> str:
    """Convert a training example to a single text string."""
    if example.get("text"):
        return example["text"]
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Load model and print info, don't train")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["dataset"]

    print(f"Loading model: {model_cfg['name']}")
    print(f"  max_seq_length: {model_cfg['max_seq_length']}")
    print(f"  load_in_4bit: {model_cfg['load_in_4bit']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )

    # Print model architecture for LoRA target verification
    print("\n=== Model Modules (verify LoRA targets) ===")
    for name, _ in model.named_modules():
        if any(t in name for t in lora_cfg["target_modules"]):
            print(f"  {name}")
    print("=== End Modules ===\n")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # Configure LoRA via unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=lora_cfg["target_modules"],
        lora_dropout=float(lora_cfg["dropout"]),
    )

    # Load and format dataset
    print(f"Loading training data: {data_cfg['train']}")
    train_examples = load_jsonl(data_cfg["train"])
    print(f"  {len(train_examples)} training examples")

    eval_examples = load_jsonl(data_cfg["eval"])
    print(f"  {len(eval_examples)} eval examples")

    train_texts = [format_example(e) for e in train_examples]
    eval_texts = [format_example(e) for e in eval_examples]

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    # SFTConfig replaces TrainingArguments in trl 0.24+
    os.makedirs(args.output, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        num_train_epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        bf16=bool(train_cfg["bf16"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["save_steps"]),
        report_to="none",
        dataset_text_field="text",
        max_length=int(model_cfg["max_seq_length"]),
        packing=False,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

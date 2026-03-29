#!/usr/bin/env python3
"""Fine-tune a model with LoRA for Go code generation.

Uses standard transformers + peft + trl. Tested with trl 0.12 + torch 2.4.
Optional unsloth support for faster training on cloud GPUs.

Usage:
    python training/train.py --config training/config/nemotron-cascade-2.yaml
    python training/train.py --config training/config/tinyllama-test.yaml
"""
import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


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
    if example.get("text"):
        return example["text"]
    inst = example.get("instruction", "")
    out = example.get("output", "")
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


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

    # Load model
    name = model_cfg["name"]
    load_4bit = bool(model_cfg.get("load_in_4bit", False))
    print(f"Loading model: {name}")
    print(f"  max_seq_length: {model_cfg['max_seq_length']}")
    print(f"  load_in_4bit: {load_4bit}")

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=load_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print LoRA target modules found in model
    print("\n=== Model Modules (verify LoRA targets) ===")
    for mod_name, _ in model.named_modules():
        if any(t in mod_name for t in lora_cfg["target_modules"]):
            print(f"  {mod_name}")
    print("=== End Modules ===\n")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # Apply LoRA
    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=lora_cfg["target_modules"],
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading training data: {data_cfg['train']}")
    train_examples = load_jsonl(data_cfg["train"])
    print(f"  {len(train_examples)} training examples")

    eval_examples = load_jsonl(data_cfg["eval"])
    print(f"  {len(eval_examples)} eval examples")

    train_texts = [format_example(e) for e in train_examples]
    eval_texts = [format_example(e) for e in eval_examples]

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    # Training arguments
    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = not use_bf16

    os.makedirs(args.output, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        num_train_epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["save_steps"]),
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=int(model_cfg["max_seq_length"]),
        packing=False,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

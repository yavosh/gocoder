#!/usr/bin/env python3
"""Validate the training pipeline with 2 steps on any GPU.

Run this before committing to a full training run.

Usage:
    python training/validate.py --config training/config/tinyllama-test.yaml
"""
import argparse
import json
import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    lc = cfg["lora"]
    tc = cfg["training"]
    dc = cfg["dataset"]

    # 1. Load model
    print(f"1. Loading model: {mc['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        mc["name"],
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=bool(mc.get("load_in_4bit", False)),
    )
    tokenizer = AutoTokenizer.from_pretrained(mc["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   OK — {model.device}")

    # 2. Apply LoRA
    print("2. Applying LoRA")
    lora_config = LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        target_modules=lc["target_modules"],
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("   OK")

    # 3. Load 10 examples
    print(f"3. Loading data from {dc['train']}")
    examples = []
    with open(dc["train"]) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            d = json.loads(line)
            if d.get("text"):
                examples.append(d["text"])
            else:
                inst = d.get("instruction", "")
                out = d.get("output", "")
                examples.append(f"### Instruction:\n{inst}\n\n### Response:\n{out}")

    train_ds = Dataset.from_dict({"text": examples[:8]})
    eval_ds = Dataset.from_dict({"text": examples[8:]})
    print(f"   OK: {len(train_ds)} train, {len(eval_ds)} eval")

    # 4. Build training args
    print("4. Building TrainingArguments")
    training_args = TrainingArguments(
        output_dir="/tmp/gocoder-validate",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=2,
        learning_rate=float(tc["learning_rate"]),
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        report_to="none",
        gradient_checkpointing=False,
    )
    print("   OK")

    # 5. Build trainer
    print("5. Building SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
        args=training_args,
    )
    print("   OK")

    # 6. Train 2 steps
    print("6. Running 2 training steps...")
    result = trainer.train()
    print(f"   OK — loss: {result.training_loss:.4f}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   VRAM used: {allocated:.1f} GB")

    print("\n=== VALIDATION PASSED ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the entire training pipeline with 1 step. Run on GPU."""
import json
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main():
    # 1. Load config
    with open("/workspace/training/config/nemotron-cascade-2.yaml") as f:
        cfg = yaml.safe_load(f)
    print("1. Config loaded OK")
    lr = float(cfg["training"]["learning_rate"])
    print(f"   learning_rate: {lr} (type: {type(lr).__name__})")

    # 2. Load model (cached from dry run)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=int(cfg["model"]["max_seq_length"]),
        load_in_4bit=bool(cfg["model"]["load_in_4bit"]),
    )
    print("2. Model loaded OK")

    # 3. Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=float(cfg["lora"]["dropout"]),
    )
    print("3. LoRA applied OK")

    # 4. Load small dataset (10 examples)
    examples = []
    with open("/workspace/data/train.jsonl") as f:
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

    train_dataset = Dataset.from_dict({"text": examples[:8]})
    eval_dataset = Dataset.from_dict({"text": examples[8:]})
    print(f"4. Dataset loaded OK: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # 5. Construct SFTConfig
    sft_config = SFTConfig(
        output_dir="/workspace/test-validate",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        max_steps=1,
        learning_rate=lr,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
        gradient_checkpointing=False,
    )
    print("5. SFTConfig constructed OK")

    # 6. Construct trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    print("6. SFTTrainer constructed OK")

    # 7. Run 1 training step
    print("7. Running 1 training step...")
    trainer.train()
    print("   Step completed OK")

    print("\n=== ALL VALIDATION PASSED ===")


if __name__ == "__main__":
    main()

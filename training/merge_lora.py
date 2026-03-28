#!/usr/bin/env python3
"""Merge LoRA adapter weights back into the base model.

Usage:
    python training/merge_lora.py --base nvidia/Nemotron-Cascade-2-30B-A3B --adapter ./output
"""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", default="./merged", help="Output directory")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output}")
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()

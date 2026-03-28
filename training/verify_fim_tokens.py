#!/usr/bin/env python3
"""Verify FIM special tokens for Nemotron-Cascade-2.

Run this BEFORE formatting the dataset. It prints the model's actual
FIM tokens so you can update formatter/fim.go constants if needed.

Usage:
    python training/verify_fim_tokens.py --model nvidia/Nemotron-Cascade-2-30B-A3B
"""
import argparse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Check FIM tokens in model tokenizer")
    parser.add_argument("--model", required=True, help="Model name or path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=== Special Tokens ===")
    print(f"All special tokens: {tokenizer.all_special_tokens}")
    print(f"Additional special tokens: {tokenizer.additional_special_tokens}")

    # Look for FIM tokens
    fim_candidates = ["fim_prefix", "fim_suffix", "fim_middle",
                      "fill", "prefix", "suffix", "middle"]
    found = {}
    for token in tokenizer.all_special_tokens + tokenizer.additional_special_tokens:
        lower = token.lower()
        for candidate in fim_candidates:
            if candidate in lower:
                found[candidate] = token

    if found:
        print("\n=== FIM Tokens Found ===")
        for name, token in found.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name}: {token!r} (id={token_id})")
    else:
        print("\n!!! No FIM tokens found in vocabulary !!!")
        print("You will need to add FIM tokens and resize the embedding layer.")
        print("Add this to train.py before training:")
        print('  tokenizer.add_special_tokens({"additional_special_tokens": ')
        print('    ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]})')
        print('  model.resize_token_embeddings(len(tokenizer))')


if __name__ == "__main__":
    main()

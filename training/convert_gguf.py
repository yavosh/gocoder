#!/usr/bin/env python3
"""Convert merged HuggingFace model to GGUF and quantize.

Requires llama.cpp tools (convert_hf_to_gguf.py and llama-quantize) to be available.

Usage:
    python training/convert_gguf.py --input ./merged --output ./go-nemotron-Q4_K_M.gguf --quant Q4_K_M
"""
import argparse
import shutil
import subprocess
import sys


def find_tool(name: str) -> str | None:
    """Find a tool in PATH or common llama.cpp locations."""
    path = shutil.which(name)
    if path:
        return path
    # Check common llama.cpp build locations
    for candidate in [f"./llama.cpp/{name}", f"../llama.cpp/{name}", f"./llama.cpp/build/bin/{name}"]:
        if shutil.which(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("--input", required=True, help="Path to merged HF model")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")
    args = parser.parse_args()

    # Find conversion script
    convert_script = find_tool("convert_hf_to_gguf.py")
    if not convert_script:
        # Try as a Python module
        convert_script = shutil.which("python3")
        if convert_script:
            # Assume llama-cpp-python is installed
            convert_cmd = [convert_script, "-m", "llama_cpp.convert"]
        else:
            print("Error: convert_hf_to_gguf.py not found.", file=sys.stderr)
            print("Install llama.cpp or add it to PATH.", file=sys.stderr)
            sys.exit(1)
    else:
        convert_cmd = ["python3", convert_script]

    # Find quantize tool
    quantize_tool = find_tool("llama-quantize")
    if not quantize_tool:
        print("Error: llama-quantize not found.", file=sys.stderr)
        print("Build llama.cpp or add it to PATH.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Convert to GGUF (fp16)
    fp16_path = args.output.replace(".gguf", "-fp16.gguf")
    print(f"Converting {args.input} to GGUF (fp16)...")
    subprocess.run(
        [*convert_cmd, args.input, "--outfile", fp16_path],
        check=True,
    )

    # Step 2: Quantize
    print(f"Quantizing to {args.quant}...")
    subprocess.run(
        [quantize_tool, fp16_path, args.output, args.quant],
        check=True,
    )

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()

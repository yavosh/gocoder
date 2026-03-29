#!/bin/bash
set -euo pipefail

echo "=== GoCoder Cloud Training Setup ==="

# Install monitoring
apt update -qq && apt install -y -qq htop btop tmux

# Install Python deps (pinned versions that work)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.3 trl==0.12.2 peft==0.13.2 datasets==3.0.0 \
    accelerate==1.2.1 bitsandbytes==0.44.1 pyyaml sentencepiece protobuf

# Verify dataset
echo ""
echo "=== Dataset ==="
wc -l /workspace/data/train.jsonl /workspace/data/eval.jsonl

# Validate with 2 steps
echo ""
echo "=== Validation (2 training steps) ==="
python /workspace/training/validate.py --config /workspace/training/config/nemotron-cascade-2.yaml

echo ""
echo "=== Setup Complete ==="
echo "To start training in tmux:"
echo "  tmux new -s train"
echo "  python /workspace/training/train.py \\"
echo "    --config /workspace/training/config/nemotron-cascade-2.yaml \\"
echo "    --output /workspace/run-01 2>&1 | tee /workspace/run-01.log"

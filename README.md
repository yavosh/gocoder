# GoCoder

Fine-tune open-source LLMs for Go development. Extract high-quality training data from Go codebases, fine-tune Nemotron-Cascade-2 with LoRA, evaluate results, and serve locally via ollama.

## Quick Start

```bash
# Build
make build

# Extract training data from a Go codebase
bin/gocoder pipeline build --dir /path/to/go/repo --output data/output

# Run eval against a model
bin/gocoder eval run --endpoint http://localhost:11434/v1 --model go-nemotron --prompts eval/prompts

# Start the model routing proxy
bin/gocoder serve --config serving/config.yaml
```

## Architecture

Two parallel tracks:

**Track A: Cloud (start now)** — build dataset, train on cloud H100, iterate on quality

```
sources.yaml -> pipeline build -> train.jsonl + eval.jsonl
                                       |
                               cloud GPU (H100)
                                       |
                               train -> merge -> quantize -> go-nemotron.gguf
                                       |
                               eval -> compare -> iterate
```

**Track B: Local (when hardware arrives)** — deploy model, serve via router, use with opencode

```
ollama (go-nemotron, qwen3, qwen3.5) -> gocoder serve -> opencode
```

## Components

### Dataset Pipeline (`gocoder pipeline build`)

Extracts Go functions from source code using `go/parser`, filters generated code and duplicates, formats as FIM (autocomplete) or instruction (generation) pairs.

```bash
bin/gocoder pipeline build \
  --dir /path/to/repo \
  --output data/output \
  --min-lines 3 \
  --fim-ratio 0.4 \
  --seed 42
```

**Sources** configured in `data/sources.yaml`:
- Go stdlib, CockroachDB, Kubernetes, Prometheus, Hashicorp, CoreDNS, etc.
- Go blog, Effective Go, Uber/Google style guides, Dave Cheney, Eli Bendersky
- Personal repos (oversampled) — supports both `url:` (remote) and `dir:` (local paths)

### Training Scripts (`training/`)

Python scripts for LoRA fine-tuning. Uses standard transformers + peft + trl (no unsloth dependency).

| Script | Purpose |
|---|---|
| `train.py` | LoRA fine-tuning with trl SFTTrainer |
| `validate.py` | Run 2 training steps to verify pipeline before full run |
| `merge_lora.py` | Merge adapter into base model |
| `convert_gguf.py` | Convert to GGUF + quantize |
| `verify_fim_tokens.py` | Check FIM tokens before training |
| `cloud-setup.sh` | One-shot cloud GPU setup (deps, cache, validation) |

**Tested stack:** torch 2.4 + trl 0.12.2 + peft 0.13.2 + transformers 4.46.3

```bash
# Validate locally first (any GPU)
python training/validate.py --config training/config/tinyllama-test.yaml

# Full training on cloud
python training/train.py --config training/config/nemotron-cascade-2.yaml --output ./run-01
```

### Evaluation Harness (`gocoder eval`)

29 Go-specific prompts across 9 categories. Automated scoring (compiles, vet clean) plus LLM-as-judge for quality assessment.

| Category | Weight | Focus |
|---|---|---|
| error_handling | 0.15 | fmt.Errorf, sentinel errors, errors.Is/As |
| concurrency | 0.15 | goroutines, channels, context cancellation |
| http_middleware | 0.10 | handler chains, auth, rate limiting |
| interfaces | 0.10 | small interfaces, io.Reader patterns |
| testing | 0.10 | table-driven, httptest, helpers |
| context_propagation | 0.10 | threading, timeouts, values |
| package_design | 0.10 | internal/, naming, dependency direction |
| struct_methods | 0.10 | receivers, embedding, constructors |
| idiom | 0.10 | naming, zero values, comma-ok |

### Model Router (`gocoder serve`)

Thin OpenAI-compatible proxy that routes requests to different ollama models based on task type.

```yaml
# serving/config.yaml
routing:
  - match: { path: "/v1/completions" }
    model: autocomplete          # qwen3:30b-a3b (~196 tok/s)
  - match: { model: "go-*" }
    model: codegen               # go-nemotron (fine-tuned)
  - match: { default: true }
    model: codegen
```

## Local Testing with GTX 1080

The training pipeline can be validated on older GPUs before spending on cloud. Tested on a GTX 1080 (8GB VRAM, CUDA compute 6.1).

### Constraints

- **Torch 2.4** — PyTorch 2.10+ dropped Pascal support (sm_61). Pin to torch 2.4.
- **trl 0.12** — Must match torch 2.4 compatible stack. trl 0.24+ requires torch 2.10+.
- **No unsloth** — Requires torch 2.10+. Use standard transformers + peft instead.
- **No bf16** — Pascal lacks native bf16 support. Use fp16.
- **4-bit quantization optional** — TinyLlama (1.1B) fits in fp16 without quantization. Larger models need 4-bit.

### Setup (WSL2 on Windows)

```bash
# Install CUDA toolkit (driver comes from Windows)
sudo apt install cuda-libraries-12-4 cuda-compiler-12-4 cuda-cudart-12-4

# nvidia-smi is at a different path in WSL
alias nvidia-smi=/usr/lib/wsl/lib/nvidia-smi

# Create venv with compatible stack
python3 -m venv ~/gocoder-env
source ~/gocoder-env/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.3 trl==0.12.2 peft==0.13.2 datasets==3.0.0 \
    accelerate==1.2.1 bitsandbytes==0.44.1 pyyaml sentencepiece protobuf

# Validate
python training/validate.py --config training/config/tinyllama-test.yaml
```

### Performance

| GPU | TinyLlama 1.1B (11K examples, 1 epoch) | Nemotron-Cascade-2 30B |
|---|---|---|
| GTX 1080 (8GB) | ~31 hrs | Won't fit |
| RTX 4090 (24GB) | ~10 min | ~1.5-2 hrs (4-bit) |
| H100 (80GB) | ~2 min | ~30-45 min (bf16) |

The 1080 is for **validation only** — confirm the script works before renting cloud GPUs.

## Cloud Training

### RunPod Setup

1. Create pod: H100 SXM 80GB, PyTorch template
2. Upload data + scripts
3. Run setup:

```bash
# From your Mac
scp -P <port> data/output/*.jsonl root@<ip>:/workspace/data/
scp -P <port> -r training/ root@<ip>:/workspace/training/

# On the pod
bash /workspace/training/cloud-setup.sh

# In tmux
tmux new -s train
python /workspace/training/train.py \
  --config /workspace/training/config/nemotron-cascade-2.yaml \
  --output /workspace/run-01 2>&1 | tee /workspace/run-01.log
```

### Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `learning_rate: '<' not supported` | YAML parses `2e-4` as string | Explicit `float()` cast in train.py |
| `gradient checkpointing not supported` | Nemotron-Cascade-2 architecture limitation | Set `gradient_checkpointing=False` |
| `no kernel image available` | PyTorch too new for GPU | Pin torch to version supporting your GPU's compute capability |
| Windows sleep kills training | Power management suspends GPU | `powercfg /change standby-timeout-ac 0` |

## Project Structure

```
gocoder/
├── cmd/gocoder/          # CLI entrypoint
├── internal/
│   ├── config/           # sources.yaml parser
│   ├── collector/        # clone repos, fetch articles
│   ├── extractor/        # Go AST extraction, HTML prose extraction
│   ├── filter/           # generated code, dedup, quality gates
│   ├── formatter/        # FIM + instruction format conversion
│   ├── pipeline/         # orchestrates collect -> extract -> filter -> format
│   ├── eval/             # runner, scorer, compare, LLM judge
│   └── router/           # OpenAI-compatible reverse proxy
├── eval/prompts/         # YAML eval prompts (9 categories)
├── serving/              # ollama config + Modelfiles
├── training/             # Python fine-tuning scripts
│   └── config/           # Model-specific training configs
└── data/                 # sources.yaml + dataset output
```

## Hardware Target

RTX 4090 24GB + 128GB DDR5 RAM (~$3,200). Runs 30B MoE models in VRAM, frontier models via CPU offloading.

| Model | Use | tok/s |
|---|---|---|
| Qwen 3 30B-A3B | Fast autocomplete | ~196 |
| go-nemotron (fine-tuned) | Go code generation | ~80-100 |
| Qwen 3.5 27B | Multi-file refactoring | ~25-40 |

## Development

```bash
make build    # Build binary
make test     # Run tests
make lint     # Run go vet + staticcheck
make clean    # Remove build artifacts
```

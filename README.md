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
sources.yaml → pipeline build → train.jsonl + eval.jsonl
                                      ↓
                              cloud GPU (H100)
                                      ↓
                              train → merge → quantize → go-nemotron.gguf
                                      ↓
                              eval → compare → iterate
```

**Track B: Local (when hardware arrives)** — deploy model, serve via router, use with opencode

```
ollama (go-nemotron, qwen3, qwen3.5) → gocoder serve → opencode
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
- Personal repos (oversampled)

### Training Scripts (`training/`)

Python scripts for cloud GPU fine-tuning:

| Script | Purpose |
|---|---|
| `train.py` | LoRA fine-tuning with unsloth/trl |
| `merge_lora.py` | Merge adapter into base model |
| `convert_gguf.py` | Convert to GGUF + quantize |
| `verify_fim_tokens.py` | Check FIM tokens before training |

```bash
# On cloud GPU
python training/train.py --config training/config/nemotron-cascade-2.yaml
python training/merge_lora.py --base nvidia/Nemotron-Cascade-2-30B-A3B --adapter ./output
python training/convert_gguf.py --input ./merged --output go-nemotron-Q4_K_M.gguf --quant Q4_K_M
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
│   ├── pipeline/         # orchestrates collect → extract → filter → format
│   ├── eval/             # runner, scorer, compare, LLM judge
│   └── router/           # OpenAI-compatible reverse proxy
├── eval/prompts/         # YAML eval prompts (9 categories)
├── serving/              # ollama config + Modelfiles
├── training/             # Python fine-tuning scripts
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

# GoCoder Design Spec

## Overview

Build a Go-specialized LLM fine-tuning pipeline, evaluation harness, and local serving stack. Fine-tune Nemotron-Cascade-2 (30B MoE, 3B active params) on high-quality Go code and prose, serve it locally on an RTX 4090 (24GB), and integrate with opencode for daily use.

Two parallel tracks: cloud training starts immediately, local infrastructure is built when hardware arrives.

## Decisions

- **Hardware:** RTX 4090 24GB + 128GB DDR5 RAM (~$3,200)
- **Base model:** Nemotron-Cascade-2 (30B MoE, 3B active params)
- **Training method:** bf16 LoRA (not QLoRA — MoE compatibility issues)
- **Training compute:** Cloud H100 80GB (primary), A100 80GB (fallback)
- **Pre-hardware serving:** Hugging Face Inference Endpoints
- **Editor integration:** opencode (terminal AI coding tool, OpenAI-compatible)
- **Approach:** Parallel tracks — cloud training now, local stack when hardware arrives

## Project Structure

```
gocoder/
├── CLAUDE.md
├── Makefile                          # Top-level orchestration
├── pipeline/                         # Dataset pipeline (Go CLI, invoked via `gocoder pipeline build`)
│   ├── cmd/pipeline/main.go
│   ├── internal/
│   │   ├── collector/                # Clone repos, fetch articles
│   │   ├── extractor/                # Extract Go code, parse articles
│   │   ├── filter/                   # Remove generated code, dedup, quality gate
│   │   └── formatter/                # Convert to FIM + instruction JSONL
│   ├── go.mod
│   └── go.sum
├── training/                         # Fine-tuning scripts (Python)
│   ├── train.py                      # Main training script (unsloth + trl)
│   ├── merge_lora.py                 # Merge adapter with base
│   ├── convert_gguf.py               # Convert to GGUF + quantize
│   ├── config/
│   │   └── nemotron-cascade-2.yaml   # Training hyperparams
│   └── requirements.txt
├── eval/                             # Evaluation harness (Go + Python, invoked via `gocoder eval run`)
│   ├── prompts/                      # Go coding prompts (YAML)
│   ├── judge/                        # Scoring logic
│   ├── cmd/eval/main.go              # Run eval suite against ollama API
│   └── results/                      # Scores per run (gitignored)
├── serving/                          # Model routing + serving config
│   ├── router/                       # Request router (Go, thin proxy)
│   ├── models/                       # Modelfiles for ollama
│   └── config.yaml                   # Model routing rules
├── data/                             # Dataset sources config (gitignored outputs)
│   ├── sources.yaml                  # Repos, articles, blogs to collect
│   └── .gitignore
├── docs/
│   └── superpowers/specs/
└── tasks/
```

### Language Choices

- **Dataset pipeline:** Go — AST-level extraction via `go/parser`, maintainable by the user
- **Training scripts:** Python — required by the ecosystem (unsloth, trl, transformers)
- **Eval harness:** Go — CLI that calls ollama-compatible API, Python helper for LLM-as-judge
- **Serving router:** Go — thin HTTP proxy, OpenAI-compatible

## Dataset Pipeline

Go CLI tool that takes `sources.yaml` and produces `train.jsonl` + `eval.jsonl`.

### Sources

Three tiers:

1. **Articles/blogs** — Go best practices, style guides, reasoning about code (instruction format)
2. **Open-source repos** — canonical and production-quality Go (FIM + instruction)
3. **Personal repos** — 4-5 repos, oversampled to learn the user's patterns (FIM + instruction)

### Sources Config

```yaml
# data/sources.yaml
repos:
  # Open source
  - url: https://github.com/golang/go
    path: src/
    weight: 1.5
  - url: https://github.com/cockroachdb/cockroach
  - url: https://github.com/kubernetes/kubernetes
  - url: https://github.com/prometheus/prometheus
  - url: https://github.com/hashicorp/consul
  - url: https://github.com/hashicorp/vault
  - url: https://github.com/hashicorp/terraform
  - url: https://github.com/etcd-io/etcd
  - url: https://github.com/charmbracelet/bubbletea
  - url: https://github.com/sqlc-dev/sqlc
  - url: https://github.com/coredns/coredns

  # Personal repos (4-5)
  - url: https://github.com/yavosh/repo1
    weight: 2.0
  # ...

articles:
  # Official
  - url: https://go.dev/blog/
    type: blog_index
  - url: https://go.dev/doc/effective_go
    type: single_page

  # Style guides
  - url: https://github.com/uber-go/guide
    type: markdown_repo
  - url: https://google.github.io/styleguide/go/
    type: sitemap

  # Deep dives
  - url: https://dave.cheney.net/
    type: blog_index
  - url: https://eli.thegreenplace.net/tag/go
    type: blog_index

  # Custom bookmarks
  - path: data/bookmarks.txt
    type: url_list
```

### Pipeline Stages

```
sources.yaml -> collect -> extract -> filter -> format -> train.jsonl + eval.jsonl
```

1. **Collect** — Clone repos (`--depth 1`), fetch articles/blogs as markdown
2. **Extract** — Parse Go files with `go/parser`, extract functions/methods/types with doc comments. Convert articles HTML to markdown, extract code blocks with surrounding prose
3. **Filter** — Remove generated code (protobuf, mocks, wire, `//go:generate`), remove vendor/, dedup (exact + near-duplicate via MinHash), remove files < 20 lines, quality gate (`go vet`)
4. **Format** — Convert to training format:
   - **FIM** from code: split functions at random points for autocomplete
   - **Instruction** from code: doc comment + signature -> implementation
   - **Instruction** from articles: prose explanation <-> code example
   - **Instruction** from issue/PR pairs: issue description -> fix diff
5. **Split** — 90/10 train/eval, stratified by category

### Design Points

- `go/parser` gives AST-level extraction — individual functions with receivers, doc comments, and imports
- Articles become instruction pairs for teaching *why* patterns are preferred, not just *what* they look like
- Weights in `sources.yaml` control oversampling
- Pipeline is idempotent — deterministic hashing for dedup, fixed random seed for splits
- Single command: `pipeline build`
- **Dataset size target:** Minimum 10K examples to start training run 1, target 50K-100K for final runs
- **FIM tokens:** Must be verified against Nemotron-Cascade-2's actual special tokens before formatting. If the model doesn't have FIM tokens in its vocabulary, they need to be added with embedding layer resize

## Training & Iteration Loop

### Cloud Setup

H100 80GB as primary (Vast.ai or RunPod), A100 80GB as fallback.

- ~$5-12 per run (2-4 hours on H100, varies with dataset size)
- 8-14 runs total: ~$40-150
- Starts immediately — no hardware dependency
- **VRAM note:** 30B params in bf16 = ~60GB model weights. H100 80GB should fit with batch_size=4 and seq_len=4096, but may need to drop to batch_size=1-2 if OOM. LoRA adapters and optimizer states are small (~1-2GB)

### Training Config

```yaml
# training/config/nemotron-cascade-2.yaml
model:
  name: nvidia/Nemotron-Cascade-2-30B-A3B
  max_seq_length: 4096
  load_in_4bit: false

lora:
  r: 16
  alpha: 16
  # NOTE: These are placeholder targets for a standard transformer.
  # MoE models have different layer names (e.g., experts.0.gate_proj).
  # Run `model.named_modules()` on the loaded model to discover actual
  # layer names before training. Unsloth may handle this automatically.
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  dropout: 0.05

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  epochs: 2
  learning_rate: 2e-4
  bf16: true
  save_steps: 500

dataset:
  train: data/train.jsonl
  eval: data/eval.jsonl
  fim_ratio: 0.4
```

### Iteration Protocol

Each run follows: adjust config/data -> train on cloud -> download adapter -> merge -> quantize -> eval -> compare.

**Artifact versioning:** Tag each run's adapter, GGUF, and eval results with the run number (e.g., `run-05/`). Keep the best-known-good model tagged so you can roll back if a run regresses.

| Run | Goal | Change | Decision Gate |
|---|---|---|---|
| 1 | Sanity check | Default config, full dataset | Loss converges? Any crashes? |
| 2-3 | Data quality | Remove bad examples found in eval | Scores improving vs baseline? |
| 4-5 | Hyperparams | Try r=32, lr=1e-4, epochs=3 | Diminishing returns? Pick best |
| 6-7 | Format balance | Adjust FIM/instruction ratio, article weight | FIM vs instruction quality |
| 8-10 | Data expansion | Add more of what works, remove what hurts | Convergence on best mix |
| 11-14 | Final polish | Best config, cleaned dataset, 3 epochs | Ship if scores plateau |

### Cloud Workflow (per run)

```bash
# Upload dataset
rsync -avP data/{train,eval}.jsonl cloud:/workspace/data/

# Train
python training/train.py --config training/config/nemotron-cascade-2.yaml

# Merge + convert
python training/merge_lora.py --base nvidia/Nemotron-Cascade-2-30B-A3B --adapter /workspace/output/checkpoint-best
python training/convert_gguf.py --input /workspace/merged --output /workspace/go-nemotron-Q4_K_M.gguf --quant Q4_K_M

# Download
rsync -avP cloud:/workspace/go-nemotron-Q4_K_M.gguf models/

# Eval
gocoder eval run --model models/go-nemotron-Q4_K_M.gguf --baseline results/baseline.json
```

## Evaluation Harness

Go CLI that runs Go-specific coding prompts against any ollama-compatible API and scores results.

### Prompt Categories

| Category | Weight | Description |
|---|---|---|
| error_handling | 0.15 | fmt.Errorf wrapping, sentinel errors, errors.Is/As |
| concurrency | 0.15 | goroutines, channels, sync primitives, context cancellation |
| http_middleware | 0.10 | handler chains, request/response manipulation |
| interfaces | 0.10 | small interfaces, implicit satisfaction, io.Reader patterns |
| testing | 0.10 | table-driven tests, testify, httptest |
| context_propagation | 0.10 | context.Context threading, timeouts, values |
| package_design | 0.10 | internal/, naming, dependency direction |
| struct_methods | 0.10 | pointer vs value receivers, embedding, composition |
| idiom | 0.10 | naming conventions, zero values, comma-ok, range |

### Scoring Modes

**Automated (every run):**
- Compiles (`go build`)
- Vet clean (`go vet`)
- Staticcheck passes
- Tests pass (if applicable)
- Composite weighted score

**LLM-as-judge (milestone runs 1, 7, 14):**
- Send prompt + output to frontier model (Opus via API)
- Score 1-5: correctness, idiom, simplicity, completeness
- Validates automated scores correlate with quality

### Eval CLI

```bash
gocoder eval run --model go-nemotron --output results/run-05.json
gocoder eval run --endpoint http://cloud:11434 --model go-nemotron --output results/run-05.json
gocoder eval compare --baseline results/baseline.json --candidate results/run-05.json
gocoder eval judge --input results/run-05.json --judge claude-opus
```

## Serving & Model Routing

Thin Go HTTP proxy in front of ollama. Routes requests to the right model based on task type.

### Routing Config

```yaml
# serving/config.yaml
listen: :8080
ollama: http://localhost:11434

models:
  autocomplete:
    model: qwen3:30b-a3b
  codegen:
    model: go-nemotron
  reasoning:
    model: qwen3.5:27b
  frontier:
    model: kimi-k2.5

routing:
  - match: { path: "/v1/completions" }
    model: autocomplete
  - match: { model: "go-*" }
    model: codegen
  - match: { model: "reason" }
    model: reasoning
  - match: { model: "frontier" }
    model: frontier
  - match: { default: true }
    model: codegen
```

### Design Points

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`)
- Transparent proxy — swaps model field, passes everything else through
- Streaming SSE pass-through, no buffering
- Optional model preloading on startup
- Health endpoint (`/health`) and Prometheus metrics (`/metrics`)
- Localhost only, no auth
- Router only depends on the OpenAI-compatible API subset (`/v1/*`), not ollama-specific APIs. This means it works with HF Inference Endpoints or any OpenAI-compatible backend without changes

### Modelfile

```dockerfile
FROM ~/models/go-nemotron-Q4_K_M.gguf
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM """You are an expert Go developer. Write idiomatic, simple Go code.
Use proper error handling with fmt.Errorf and %w. Propagate context.Context.
Prefer composition over inheritance. Keep interfaces small."""
```

## OpenCode Integration

OpenCode speaks OpenAI-compatible API. Point it at the router.

### Configuration

```json
{
  "provider": {
    "name": "gocoder",
    "type": "openai",
    "base_url": "http://localhost:8080/v1",
    "models": {
      "default": "go-nemotron",
      "fast": "qwen3:30b-a3b",
      "reasoning": "reason"
    }
  }
}
```

### Connection Flow

```
opencode -> router (:8080) -> ollama (:11434) -> GPU
```

### Pre-Hardware (Remote)

Before hardware arrives, point opencode directly at HF Inference Endpoint:

```json
{
  "provider": {
    "name": "gocoder",
    "type": "openai",
    "base_url": "https://your-endpoint.hf.space/v1"
  }
}
```

Or point the router at HF instead of local ollama:

```yaml
ollama: https://your-endpoint.hf.space/v1
```

### What We Don't Build

- No custom opencode plugin — it already supports OpenAI-compatible APIs
- No custom LSP server — opencode handles that
- No inline autocomplete integration (future add-on)

## Architecture Diagram

### Track A: Cloud (Start Now)

```
sources.yaml
    |
    v
pipeline build            <- Go CLI, runs on Mac
    |
    +-- train.jsonl
    +-- eval.jsonl
          |
          v
    Upload to cloud GPU (H100)
          |
          v
    train.py (unsloth + trl + peft)
          |
          +-- LoRA adapter
          v
    merge_lora.py -> convert_gguf.py -> go-nemotron-Q4_K_M.gguf
          |
          +-- Upload to Hugging Face (serving + versioning)
          +-- Download locally (when hardware arrives)
          |
          v
    eval run --endpoint <HF or cloud>
          |
          v
    Compare to baseline -> decide next iteration
```

### Track B: Local (When Hardware Arrives)

```
ollama create go-nemotron -f Modelfile    <- pull GGUF from HF or local
    |
    v
ollama (localhost:11434)
    |  +-- qwen3:30b-a3b        (autocomplete)
    |  +-- go-nemotron           (fine-tuned Go)
    |  +-- qwen3.5:27b           (reasoning)
    |  +-- kimi-k2.5             (frontier, CPU offload)
    |
    v
gocoder serve (localhost:8080)   <- Go router proxy
    |
    v
opencode                         <- terminal AI coding tool
```

## Timeline

| When | What | Depends On |
|---|---|---|
| Week 1 | Build dataset pipeline, collect sources, order hardware | Nothing |
| Week 2 | Clean/format dataset, run baseline eval on vanilla model via cloud | Pipeline done |
| Week 3 | Training runs 1-7 on cloud H100, iterate on data quality | Dataset ready |
| Week 4 | Training runs 8-14, publish best model to HF, start using via HF endpoint + opencode | Converging on quality |
| Week 5 | Hardware arrives — assemble, install OS/drivers/ollama, deploy model locally | Hardware shipped |
| Week 6 | Build router, wire up opencode to local stack, final eval on local hardware | Everything converges |

## Budget

| Category | Cost |
|---|---|
| Hardware (RTX 4090 build) | $3,000-3,300 |
| Cloud GPU training (H100, 8-14 runs) | $40-150 |
| HF Inference Endpoint (interim serving) | Variable, check current pricing |
| Electricity (ongoing) | ~$10-20/mo |
| **Total to production** | **~$3,050-3,450 + interim hosting** |

## Deliverables

1. **`gocoder pipeline build`** — Go CLI. Produces train/eval JSONL from sources.yaml
2. **`training/`** — Python scripts. Train, merge, convert, quantize. Config-driven
3. **`gocoder eval`** — Go CLI. `run`, `compare`, `judge` subcommands. Automated + LLM-as-judge
4. **`gocoder serve`** — Go HTTP proxy. Routes to right ollama model. OpenAI-compatible
5. **`go-nemotron`** — The fine-tuned model. GGUF on HF, Modelfile for ollama
6. **opencode config** — Point and shoot. Local or remote, same interface

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Fine-tuning degrades general capability | Model worse at non-Go tasks | Keep base models installed, use fine-tuned only for Go |
| Dataset quality issues | Model learns bad patterns | Manual review of 100+ samples before training |
| MoE + LoRA compatibility | Training fails or is slow | Use bf16 LoRA, rent A100/H100 for headroom |
| RTX 4090 availability/pricing | Budget overrun | Buy used, budget $1,600-2,200 |
| Model architecture changes | Better model releases mid-project | Dataset and pipeline are reusable — swap base model |
| HF endpoint latency | Slow pre-hardware experience | Acceptable for interim, local stack is the goal |

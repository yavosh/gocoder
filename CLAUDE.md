# GoCoder — Local Go-Specialized LLM Project

## Project Goal

Build a local machine for running and fine-tuning open-source LLMs specialized for Go development. The fine-tuned model should approach frontier quality for idiomatic Go coding tasks.

## Key Decisions Made

### Hardware: Two Options

- **Option A (~$3,200):** RTX 4090 24GB + 128GB DDR5 RAM. Runs 30B MoE models in VRAM, frontier models via CPU offloading (slow).
- **Option B (~$9,200):** RTX Pro 6000 Blackwell 96GB GDDR7 + 64GB DDR5 RAM. Runs 120B MoE and 70B dense models fully in VRAM. Matches single H100 performance. Rarely needs cloud APIs.
- **DGX Spark rejected:** 128GB memory but only 273 GB/s bandwidth (vs 1,792 GB/s Pro 6000). 6-7x slower inference — unusable for interactive coding.
- **RTX 4000 Ada rejected:** Only 20GB VRAM, ~30-35% compute of 4090. Can't run Nemotron-Cascade-2.
- **CPU:** Intel Core Ultra 7 270K Plus ($299) or AMD Ryzen 7 9800X3D ($400). CPU is least important component.

### Target Model for Fine-Tuning

- **Nemotron-Cascade-2** (NVIDIA, released March 20, 2026) — 30B MoE, 3B active params
- LiveCodeBench v6: 87.2 (beats Kimi K2.5, Qwen3.5-397B)
- Gold medal level on IOI, ICPC, IMO
- Fits on single RTX 4090 (Q4_K_M ~24.5GB)
- 3B active params = fast QLoRA/LoRA fine-tuning

### Model Stack (by task)

| Layer | Model | tok/s (4090) | tok/s (Pro 6000) | Use |
|---|---|---|---|---|
| Fast autocomplete | Qwen 3 30B-A3B | ~196 | ~300+ | Copilot-style |
| Code generation | Nemotron-Cascade-2 (fine-tuned) | ~80-100 | ~150+ | Go code gen |
| Software engineering | Qwen 3.5 27B | ~25-40 | ~60-100 | Multi-file, refactoring |
| Frontier (Pro 6000 only) | Nemotron 3 Super 120B | N/A | ~40-60 | SWE-bench 60.5% |
| Frontier (Pro 6000 only) | Llama 70B class | N/A | ~50-70 | Deep reasoning |

### Fine-Tuning Strategy

- **Train on cloud** (RunPod H100 80GB ~$2.69/hr) — $40-150 total for 8-14 iteration runs
- **Toolchain:** transformers + peft + trl 0.12.2 + torch 2.4 (validated locally and on cloud)
- **Unsloth:** NOT currently used — requires trl 0.24 + torch 2.10 which has breaking API changes. Can be added later for speed optimization.
- **Method:** bf16 LoRA on cloud (fp16 on local test GPU)
- **Dataset:** Own Go repos (12,521 examples from ziglu, plutus, orcan, smtpbox, mrcrawley) + open source repos planned
- **Format:** Mix of FIM (autocomplete) and instruction (generation)
- **Local validation:** GTX 1080 (8GB) with TinyLlama — validates pipeline before cloud spend

### Training Troubleshooting (Learned)

- YAML `2e-4` parsed as string by pyyaml — always cast with `float()`, `int()`, `bool()`
- Nemotron-Cascade-2 does NOT support gradient checkpointing — set `gradient_checkpointing=False`
- trl 0.24 renamed `tokenizer` to `processing_class` and `TrainingArguments` to `SFTConfig` — stick with trl 0.12.2 for now
- GTX 1080 (Pascal, sm_61) needs torch 2.4 — PyTorch 2.10+ dropped support
- Windows power management can suspend GPU mid-training — disable with `powercfg /change standby-timeout-ac 0`
- Always run `validate.py` (2 training steps) before committing to a full training run

### Open Source Frontier Landscape (March 2026)

Top open models: GLM-5, Kimi K2.5, DeepSeek-V3.2, GLM-4.7, Nemotron 3 Super, Mistral Large 3.
MoE architecture dominates — large total params, small active params.
Gap between open source and closed frontier (Opus, GPT-5) is near parity on many benchmarks.

## Files

- `tasks/local-go-llm-build.md` — Detailed 6-phase implementation plan with hardware options, model reference cards, training config, timeline, and budget

## Technical Context

- LLM inference is memory-bandwidth-bound, not compute-bound
- Top SSD: ~12-14 GB/s, DDR5: ~80-90 GB/s, HBM3e: ~3,000-5,000 GB/s
- MoE models need all experts loaded in memory even though only a subset fires per token
- CPU offloading via llama.cpp is functional but slow (limited by DDR5 bandwidth ~90 GB/s vs GDDR7 1,792 GB/s)
- HBM has higher latency than DDR5 (~100-200ns vs ~60-80ns) because it's optimized for throughput not latency — GPU programming model hides this with massive parallelism

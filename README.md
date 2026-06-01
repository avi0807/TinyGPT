# TinyGPT

A small GPT-style language model built and trained from scratch to write short children's stories. Implemented in TensorFlow/Keras with a custom transformer (RoPE attention, weight tying, KV-cache), a byte-level BPE tokenizer, and a FastAPI inference server.

**🔗 Live demo:** [avi080704-tinygpt.hf.space](https://avi080704-tinygpt.hf.space)

<img width="1902" height="965" alt="image" src="https://github.com/user-attachments/assets/0e1e2f4f-3892-48a0-87b3-473abccf4e4e" />

<p align="center">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Tokenizer](#tokenizer)
- [Positional Encoding — RoPE](#positional-encoding--rope)
- [Training](#training)
- [Dataset](#dataset)
- [Inference & Sampling](#inference--sampling)
- [KV-Cache](#kv-cache)
- [API Server](#api-server)
- [Temperature Control](#temperature-control)
- [Database](#database)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Server](#running-the-server)
- [Deployment](#deployment)
- [Training From Scratch](#training-from-scratch)
- [Results](#results)

---

## Overview

TinyGPT is a decoder-only transformer trained from scratch, without pretrained weights. Every core component — the tokenizer, attention with rotary embeddings, the training loop, the sampling code, and the KV-cache used for fast generation — is implemented directly.

The model is trained on the TinyStories dataset and learns to generate coherent, grammatically correct short stories from a prompt. The narrow, uniform story domain is what lets a model this small produce genuinely readable output.

```
Parameters:   ~50M
Architecture: GPT-style decoder-only transformer (GPT-3 inspired)
Positional:   RoPE (rotary position embeddings)
Tokenizer:    Byte-level BPE (HuggingFace tokenizers), 10k vocab
Framework:    TensorFlow 2.x / Keras (mixed bfloat16)
Dataset:      TinyStoriesV2 (noanabeshima/TinyStoriesV2)
Final loss:   ~1.42 (validation cross-entropy)
```

---

## Model Architecture

A **decoder-only transformer** following GPT-2/3 design, with RoPE instead of learned absolute positions.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `d_model` | 640 |
| `num_heads` | 10 (head_dim = 64) |
| `dff` (feed-forward dim) | 2560 |
| `num_layers` | 10 |
| `seq_len` | 256 |
| `dropout_rate` | 0.1 |
| `vocab_size` | 10000 |

### Components

**Token Embedding**
- Learned embedding matrix of shape `(vocab_size, d_model)`, initialized `N(0, 0.02)`.
- Position is handled by RoPE inside attention, so there is no separate positional embedding table.

**Transformer Blocks (×10)** — Pre-LayerNorm (GPT-3 style):

```
x → LayerNorm → MultiHeadAttention → x + residual
  → LayerNorm → FFN (GELU)          → x + residual
```

Pre-LN keeps the residual stream clean and stabilizes deep training.

**Multi-Head Attention**
- 10 heads, `head_dim = 64`.
- Causal masking; RoPE applied to Q and K before the attention scores.
- Attention-probability dropout (GPT-3 style).
- KV-cache for fast autoregressive generation.

**Feed-Forward Network**
- `d_model → dff → d_model`, **GELU** activation.

**Initialization (GPT-2/3 scheme)**
- Weights `N(0, 0.02)`; biases zero; LayerNorm γ=1, β=0.
- Residual output projections (attention `dense` and FFN `dense2`) scaled by `0.02 / sqrt(2 · num_layers)` for stability.

**Weight Tying**
- The output projection reuses the token-embedding matrix, saving `vocab_size × d_model` parameters.

**Final LayerNorm** before the output projection; logits are cast to float32.

### Parameter Count (~50M)

With `d_model=640`, `dff=2560`, `num_layers=10`, `vocab=10000`:

| Component | Params |
|-----------|-------:|
| Token embedding (tied) | 6.4M |
| 10 × transformer block (~4.9M each) | ~49M |
| Final LayerNorm | ~0.001M |
| **Total** | **~50.5M** |

Per block ≈ `12 · d_model²` (4·d² attention + 8·d² FFN) plus biases and LayerNorm.

---

## Tokenizer

A **byte-level BPE** tokenizer (HuggingFace `tokenizers`, Rust-backed) with a 10k vocabulary, trained on a sample of TinyStories.

- **Byte-level**: any unicode/whitespace round-trips cleanly; no `KeyError` on unseen characters.
- **Special tokens**: `<|endoftext|>` (id 0, EOS) and `<|unk|>` (id 1), reserved at the lowest ids.
- **Fast**: encoding the corpus is fast enough that CPU tokenization never starves the GPU during training.
- **Persistence**: saved to `saved_models/tinystories_tokenizer.json`.

A from-scratch pure-Python BPE (`BPE_tokenizer`) and a `CharTokenizer` also live in `tokenizer.py` for reference; the trained model uses `HFTokenizer`.

---

## Positional Encoding — RoPE

TinyGPT uses **Rotary Positional Embeddings (RoPE)** instead of learned or sinusoidal absolute encodings.

RoPE rotates Q and K by an angle proportional to token position:

```
q_rotated = q · cos(mθ) + rotate_half(q) · sin(mθ)
k_rotated = k · cos(mθ) + rotate_half(k) · sin(mθ)
```

When `q · k` is computed, the rotations combine to encode the **relative** distance between tokens. The sin/cos tables are precomputed once for `max_len`. During cached generation, each new token is rotated at its correct absolute position via a position offset equal to the current cache length.

Advantages: relative position directly in the attention dot product, better length behavior, no extra parameters. Used in LLaMA, Mistral, Qwen, and most modern LLMs.

---

## Training

### Optimizer

Adam with GPT-3 hyperparameters:

| Parameter | Value |
|-----------|-------|
| `beta_1` | 0.9 |
| `beta_2` | 0.95 |
| `epsilon` | 1e-8 |
| `clipnorm` | 1.0 |

### Learning Rate — Warmup + Cosine Decay

Linear warmup over the first 10% of updates to `peak_lr = 3e-4`, then cosine decay to `0.1 × peak_lr`. The schedule is driven by the number of **optimizer updates**, not micro-batches.

### Mixed Precision (bfloat16)

`mixed_bfloat16` is used. bf16 has the same exponent range as float32, so it does **not** require loss scaling (unlike float16). This avoids the `LossScaleOptimizer` machinery entirely and is stable on Ampere GPUs (RTX 3050+).

### Gradient Accumulation

To fit the model on a 6GB GPU while keeping a useful effective batch size:

```
micro_batch_size = 4
accum_steps      = 4
effective_batch  = 16
```

Gradients are summed over `accum_steps` micro-batches, then applied as a single update.

### Streaming Data Pipeline

Data is streamed and tokenized on the fly (no giant pre-tokenized array on disk):

```
HF streaming dataset → text docs → BPE tokens + <|endoftext|> boundaries
  → packed into seq_len chunks → tf.data → batched → prefetched
```

The streamer retries on transient network errors and can loop the dataset to meet a token budget larger than the dataset itself.

### Checkpointing & Resume

- Validation loss is checked every 1000 updates on an in-memory held-out set; best weights are saved to `saved_models/tinystories_model.weights.h5`.
- A `tf.train.Checkpoint` (model + optimizer + step counter) is saved to `saved_models/ckpt_tinystories/`, so a crashed run resumes from the last checkpoint.

---

## Dataset

**TinyStoriesV2** — short children's stories generated by GPT-4, designed for training and evaluating small language models.

| Property | Value |
|----------|-------|
| Source | [noanabeshima/TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2) |
| Stories | ~2.75M |
| Approx tokens | ~520–580M |
| Sequence length | 256 |

### Why TinyStories?

Small, uniform vocabulary; clear narrative structure; short sentences. A ~50M model can actually **master** this narrow distribution, which is why the output is coherent. Trained on broad web text instead, a model this size produces fluent but meaningless text — the narrow domain is the point.

---

## Inference & Sampling

Sampling strategies implemented:

- **Temperature** — divides logits before softmax. Lower = safer/more repetitive; higher = more random/creative.
- **Top-K** — keep only the K highest-probability tokens.
- **Top-P (nucleus)** — keep the smallest set of tokens whose cumulative probability exceeds `p`.
- **Repetition penalty** — already-generated tokens have their logits scaled down (penalty `1.3`) to reduce looping and repeated phrases.

`top_p` is auto-derived from the temperature (≈ 0.80 → 0.95 across the slider) so a single control adjusts everything. Generation stops early when the model emits `<|endoftext|>`.

---

## KV-Cache

Autoregressive generation uses a **KV-cache** so each new token reuses the Keys/Values computed for all previous tokens instead of recomputing the whole sequence.

Flow:
1. **Prefill**: the prompt is processed once; each layer returns its K/V (rotated by RoPE at absolute positions).
2. **Decode**: each new token is processed alone; its Q/K is rotated at `offset = current_cache_length`, concatenated onto the cached K/V, and attends over the full history.
3. Caches are passed in and **returned** from each `call` (not mutated in place), so they persist correctly across steps.

This makes generation O(n) per step instead of O(n²), and cached vs full-recompute outputs match exactly.

---

## API Server

FastAPI server exposing the model as a REST API.

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/health` | Model status, parameter count, vocab size |
| `GET` | `/temperature-info` | Slider config + green→red colour zones |
| `POST` | `/generate` | Generate a story from a prompt |
| `GET` | `/history` | Retrieve past generations |
| `DELETE` | `/history/{id}` | Delete a stored generation |

### Generate Request

```json
{
  "prompt": "Once upon a time there was a little dragon",
  "temperature": 0.5,
  "max_new_tokens": 200
}
```

`temperature` (0.1–1.5, default **0.5**) is the single creativity control. `top_p` is auto-derived from it but may be passed explicitly to override.

### Generate Response

```json
{
  "prompt": "Once upon a time there was a little dragon",
  "generated_text": "Once upon a time there was a little dragon ...",
  "temperature": 0.5,
  "temperature_label": "Safe",
  "temperature_description": "Focused and predictable. Simple, calm, easy-to-follow tales.",
  "temperature_color": "#9dc739",
  "top_p": 0.82,
  "max_new_tokens": 200,
  "response_time_ms": 812.4
}
```

---

## Temperature Control

Instead of asking users to guess what "temperature" means, the slider runs **green → red** and the API returns a plain-language label, description, and the exact colour for any value. `GET /temperature-info` returns the slider config and these zones:

| Range | Label | What it does to your stories |
|-------|-------|------------------------------|
| 0.1 – 0.5 | **Safe** | Focused and predictable. Simple, calm, easy-to-follow tales. |
| 0.5 – 0.8 | **Balanced** | A good mix of sense and surprise. Recommended for most prompts. |
| 0.8 – 1.1 | **Creative** | More imaginative and varied, with the odd unexpected twist. |
| 1.1 – 1.5 | **Wild** | Unpredictable and quirky. Fun, but may wander or stop making sense. |

In short: **lower (green) = safer and more coherent, higher (red) = more creative and more random.** The default is `0.5`.

---

## Database

Every generation is logged for monitoring and future data collection. The database is **optional**: the app defaults to a local **SQLite** file (`DATABASE_URL=sqlite:///./tinygpt.db`) and can use **PostgreSQL** by setting `DATABASE_URL`. If the database is unreachable (e.g. on ephemeral hosting), generation still works — logging is simply skipped.

### Schema — `generations`

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment |
| `prompt` | Text | Input prompt |
| `generated_text` | Text | Model output |
| `temperature` | Float | Effective sampling temperature |
| `top_p` | Float | Effective nucleus threshold |
| `max_new_tokens` | Integer | Generation length |
| `response_time_ms` | Float | Latency in ms |
| `created_at` | DateTime (indexed) | Set automatically |

Stack: `FastAPI → SQLAlchemy ORM → SQLite / psycopg2 + PostgreSQL`. Tables are created on startup.

---

## Project Structure

```
TinyGPT/
├── transformer_model/
│   ├── model.py          # GPT model, training loop, sampling, generation, data pipeline
│   ├── layers.py         # Attention, RoPE, TransformerBlock, KV-cache
│   ├── tokenizer.py      # HFTokenizer (used) + BPE_tokenizer / CharTokenizer (reference)
│   └── generation.py     # Standalone generation script
├── app/
│   ├── server.py         # FastAPI inference server
│   ├── crud.py           # DB read/write
│   ├── database.py       # SQLAlchemy engine/session
│   └── models_db.py      # generations table
├── saved_models/
│   ├── tinystories_tokenizer.json
│   ├── tinystories_model.weights.h5
│   └── ckpt_tinystories/        # resumable training checkpoints
├── index.html            # Frontend UI
├── requirements.txt          # full deps (training + serving, GPU)
├── requirements-serve.txt    # slim serving-only deps (CPU)
├── Dockerfile
├── .dockerignore
└── docker-compose.yml
```

---

## Docker

The container serves the model on CPU (no GPU needed for inference).

```bash
# build + run app and PostgreSQL together
POSTGRES_PASSWORD=yourpassword docker compose up --build
```

- App: `http://localhost:8000/`
- The image installs only `requirements-serve.txt` (CPU TensorFlow, no CUDA/training packages) and copies only the TinyStories tokenizer + weights — training checkpoints are excluded via `.dockerignore`.
- DB credentials come from env vars: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` (defaults: `avi` / `changeme` / `tiny_gpt`).

---

## Setup & Installation

### Prerequisites

- Python 3.10
- (Optional) CUDA-capable GPU. Tested on an RTX 3050 6GB under WSL2.
- PostgreSQL 14

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training/inference, `requirements.txt` pins `tensorflow[and-cuda]==2.21.0`, which bundles the matching CUDA/cuDNN wheels.

### PostgreSQL

```bash
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE tiny_gpt;"
```

Set the connection string via env var:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/tiny_gpt"
```

---

## Running the Server

```bash
source .venv/bin/activate
uvicorn app.server:app --port 8000
```

- UI: `http://localhost:8000/`
- Interactive API docs: `http://localhost:8000/docs`

To force CPU inference: `export CUDA_VISIBLE_DEVICES=-1` before launching.

---

## Deployment

The app is deployed on **Hugging Face Spaces** (free CPU tier: 2 vCPU, 16 GB RAM) as a Docker Space, live at **[avi080704-tinygpt.hf.space](https://avi080704-tinygpt.hf.space)**.

How it works:
- The repo's `README.md` front-matter (`sdk: docker`, `app_port: 7860`) tells Spaces to build and run the `Dockerfile`.
- Pushing to the Space's git repo triggers an automatic image build and redeploy.
- The model weights (~220 MB) are stored via **Git LFS**.
- Inference runs on CPU; the model loads once at startup and is reused across requests. The database falls back to SQLite (history is ephemeral on the free tier).

---

## Training From Scratch

```bash
cd transformer_model
python3 model.py
```

On the first run it trains and saves the tokenizer, then begins streaming TinyStoriesV2 and training. Progress prints per 50 updates, with validation every 1000. Best weights and resumable checkpoints are written to `saved_models/`. Re-running resumes from the last checkpoint.

Keep the process alive for long runs (`tmux`/`nohup`) and ensure the machine does not sleep.

---

## Results

Trained from scratch on TinyStoriesV2 with the ~50M configuration, validation cross-entropy reached a **final loss of ≈ 1.42**. At this level the model produces coherent short stories with a clear beginning/middle/end, dialogue, and a moral — well below the original character-level baselines and squarely in TinyStories' "coherent story" range.

**Example** (temperature 0.5, "Safe"):

> **Prompt:** "There was a boy named Avi"
>
> There was a boy named Avi. He had an incredible toy car that he loved to play with every day. One day, his mom told him they were going on a trip... They went outside and saw many fun things like birds, trees, and flowers... But then, something unexpected happened. A big wind came and blew away some of his toys!

### Notes & limitations

- **Domain-bound.** The model only knows the TinyStories world (simple words, children's-story situations). Prompts outside that domain (science, current events) won't produce sensible output — the narrow domain is what makes a 50M model coherent in the first place.
- **Uncommon names can drift.** Common TinyStories names (Lily, Tom, Ben) stay consistent. Rare names that the tokenizer splits into subwords (e.g. "Avi") can mutate across the story — a side effect of the repetition penalty discouraging exact token reuse.
- **Quality scales with training.** Output continues to sharpen as validation loss decreases.


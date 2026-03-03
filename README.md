# LilGPT

A small GPT language model trained from scratch on the TinyStories dataset. Built entirely in TensorFlow/Keras, with a custom BPE tokenizer, RoPE positional embeddings, and a FastAPI inference server backed by PostgreSQL.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Tokenizer](#tokenizer)
- [Positional Encoding — RoPE](#positional-encoding--rope)
- [Training](#training)
- [Dataset](#dataset)
- [Inference & Sampling](#inference--sampling)
- [API Server](#api-server)
- [Database](#database)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training Guidelines](#training-guidelines)
- [Results](#results)

---

## Overview

LilGPT is a decoder-only transformer language model built from scratch without using any pretrained weights or high-level model libraries. Every component — the tokenizer, attention mechanism, positional embeddings, training loop, and inference server — was implemented manually.

The model is trained on children's short stories and learns to generate coherent, grammatically correct story continuations from a prompt.

```
Parameters:   ~22M
Architecture: GPT-style decoder-only transformer
Tokenizer:    Custom BPE (Byte Pair Encoding)
Framework:    TensorFlow 2.x / Keras
Dataset:      TinyStories (roneneldan/TinyStories)
Final Loss:   ~2.32 (cross-entropy)
```

---

## Model Architecture

LilGPT uses a **decoder-only transformer** architecture following GPT-2/3 design principles.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `num_heads` | 8 |
| `dff` (feed-forward dim) | 2048 |
| `num_layers` | 8 |
| `seq_len` | 256 |
| `dropout_rate` | 0.1 |
| `vocab_size` | ~6029 |

### Components

**Token Embedding**
- Learned embedding matrix of shape `(vocab_size, d_model)`
- No separate positional embedding — position is handled by RoPE

**Transformer Blocks (×8)**

Each block uses Pre-LayerNorm (GPT-3 style) for training stability:

```
x → LayerNorm → MultiheadAttention → x + residual
  → LayerNorm → FFN               → x + residual
```

Pre-LayerNorm keeps the residual stream clean — gradients flow without being squeezed by normalization, which makes deep networks more stable than Post-LayerNorm (GPT-1/2 style).

**Multi-Head Attention**
- 8 attention heads, each with `depth = d_model / num_heads = 64`
- Causal masking via lower-triangular matrix — prevents attending to future tokens
- RoPE applied to Q and K before computing attention scores
- KV Cache implemented for efficient autoregressive generation

**Feed-Forward Network (FFN)**
- Two dense layers: `d_model → dff → d_model`
- ReLU activation
- Applied independently to each position

**Final LayerNorm**
- Applied after all transformer blocks before the output projection

**Weight Tying**
- The output projection matrix is shared with the token embedding matrix
- Reduces parameters by `vocab_size × d_model` (~3M params)
- Standard practice in GPT-1, GPT-2, GPT-3

---

## Tokenizer

A custom **Byte Pair Encoding (BPE)** tokenizer implemented from scratch.

### How BPE Works

1. Start with character-level vocabulary — every character is a token
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat for `num_merges` iterations
5. Result: a vocabulary of common subwords, words, and characters

### Configuration

| Parameter | Value |
|-----------|-------|
| `num_merges` | 6000 |
| `vocab_size` | ~6029 |
| Trained on | 100k TinyStories |

### Key Implementation Details

- **Caching**: `encode()` caches previously encoded words — critical for performance since the 100k story corpus has only ~20k unique words despite 19M total tokens
- **Persistence**: Tokenizer state saved to `tokenizer.pkl` via pickle — avoids retraining (which takes several hours) on every run
- **Encoded corpus**: `X.npy` and `Y.npy` saved after first encoding run for instant reload

---

## Positional Encoding — RoPE

LilGPT uses **Rotary Positional Embeddings (RoPE)** instead of learned or sinusoidal absolute positional encodings.

### Intuition

Standard positional encodings add position information to token vectors. RoPE instead **rotates** the Query and Key vectors by an angle proportional to their position:

```
q_rotated = q * cos(mθ) + rotate_half(q) * sin(mθ)
k_rotated = k * cos(mθ) + rotate_half(k) * sin(mθ)
```

When you compute the dot product `q · k`, the rotation angles cancel out to produce the **relative position** between tokens — naturally encoding how far apart two tokens are without needing to know their absolute positions.

### Advantages over Absolute Positional Encoding

- Encodes relative position directly in the attention dot product
- Better length generalization — can handle sequences longer than seen during training
- No extra parameters — purely mathematical transformation
- Used in LLaMA, PaLM, and most modern LLMs

### Implementation

```
build_rope_angles(seq_len, head_dim) → angles of shape (seq_len, head_dim)
rotate_half(x)                       → splits even/odd dims, applies 2D rotation
apply_rope(q, k)                     → applies rotation to Q and K before attention
```

---

## Training

### Optimizer

**Adam** with GPT-3 hyperparameters:

| Parameter | Value |
|-----------|-------|
| `beta_1` | 0.9 |
| `beta_2` | 0.95 |
| `epsilon` | 1e-8 |
| `clipnorm` | 1.0 |

Gradient clipping (`clipnorm=1.0`) prevents exploding gradients from destabilizing training on bad batches.

### Learning Rate Schedule — Warmup + Cosine Decay

```
Steps 0 → warmup_steps:      0 → peak_lr   (linear warmup)
Steps warmup_steps → total:   peak_lr → 0.1 × peak_lr  (cosine decay)
```

| Parameter | Value |
|-----------|-------|
| `peak_lr` | 3e-4 |
| `warmup_steps` | 10% of total steps |
| `min_lr` | 3e-5 (10% of peak) |

**Why warmup?** At the start of training weights are random — a high learning rate causes chaotic, destructive updates. Linear warmup lets the model stabilize before hitting peak learning rate.

**Why cosine decay?** Gradually reduces learning rate so the model makes increasingly precise weight updates as it converges, rather than overshooting the minimum.

### Mixed Precision

`mixed_float16` policy enabled via `tf.keras.mixed_precision`:

- Matrix multiplications run in **float16** — faster on tensor cores, lower memory
- Numerically sensitive ops (softmax, LayerNorm, final logits) cast to **float32**
- Reduces VRAM usage by ~40%, enabling larger batch sizes on 6GB GPU

### Early Stopping Callback

Custom `training_callback` class:

- Saves best weights to `best_model.weights.h5` whenever loss improves by `min_delta`
- Stops training if no improvement for `patience` consecutive epochs
- Plots and saves loss curve to `loss_curve.png`

| Parameter | Value |
|-----------|-------|
| `patience` | 3 |
| `min_delta` | 0.01 |

### KV Cache

Implemented for efficient autoregressive generation:

- During generation, past Key and Value tensors are cached per layer
- Each new token only attends to cached K/V instead of recomputing from scratch
- Reduces generation from O(n²) to O(n) per step
- ~4-8x speedup for long sequences

---

## Dataset

**TinyStories** — a dataset of short children's stories generated by GPT-3.5 and GPT-4, designed specifically for training and evaluating small language models.

| Property | Value |
|----------|-------|
| Source | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Stories used | 100,000 |
| Total tokens | ~19M |
| Training sequences | ~74,341 |
| Sequence length | 256 tokens |

### Why TinyStories?

- Simple, consistent vocabulary — ideal for small models
- Clear narrative structure — beginning, middle, end
- Short sentences — easier to learn grammar and coherence
- Diverse characters and scenarios within a constrained domain

### Data Pipeline

```python
raw text → BPE tokenizer → encoded_text (19M tokens)
         → sliding window (stride=seq_len) → (X, Y) sequences
         → tf.data.Dataset → shuffle → batch → prefetch
```

Sequences are created with a sliding window — input is tokens `[i:i+seq_len]`, target is `[i+1:i+seq_len+1]` (next-token prediction).

---

## Inference & Sampling

Three sampling strategies implemented:

### Temperature Sampling

Divides logits by temperature before softmax:
- `temperature < 1.0` → sharper distribution, more predictable text
- `temperature > 1.0` → flatter distribution, more creative/random text
- `temperature = 1.0` → unmodified model distribution

### Top-K Sampling

Keeps only the K highest probability tokens, zeros the rest. Prevents sampling from the long tail of unlikely tokens.

### Top-P (Nucleus) Sampling

Keeps the smallest set of tokens whose cumulative probability exceeds p. More dynamic than top-k — uses more tokens when the distribution is flat, fewer when it's peaked.

**Default generation settings:**
```python
temperature = 0.8
top_p       = 0.9
top_k       = None
```

---

## API Server

FastAPI server exposing the model as a REST API.

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serves frontend UI |
| `GET` | `/health` | Model status, param count, vocab size |
| `POST` | `/generate` | Generate text from prompt |
| `GET` | `/history` | Retrieve past generations from DB |

### Generate Request

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9
}
```

### Generate Response

```json
{
  "prompt": "Once upon a time",
  "generated_text": "Once upon a time there was...",
  "temperature": 0.8,
  "top_p": 0.9,
  "max_new_tokens": 100,
  "response_time_ms": 2341.5
}
```

---

## Database

PostgreSQL database (`lil_gpt`) stores every generation for monitoring and future fine-tuning data collection.

### Schema — `generations` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment |
| `prompt` | String | Input prompt |
| `generated_text` | String | Model output |
| `temperature` | Float | Sampling temperature |
| `top_p` | Float | Nucleus sampling threshold |
| `max_new_tokens` | Integer | Generation length |
| `response_time_ms` | Float | Latency in milliseconds |
| `created_at` | DateTime | Auto-set by PostgreSQL |

### Stack

```
FastAPI → SQLAlchemy ORM → psycopg2 driver → PostgreSQL 14
```

Tables are created automatically on server startup via `Base.metadata.create_all()`.

---

## Project Structure

```
transformers-build/
├── model.py              # GPT model, training loop, sampling functions
├── layers.py             # TransformerBlock, MultiheadAttention, RoPE
├── tokenizer.py          # BPE tokenizer implementation
├── server.py             # FastAPI inference server
├── generation.py         # Standalone generation script
├── database.py           # SQLAlchemy connection and session
├── models_db.py          # Database table definitions
├── crud.py               # Database read/write operations
├── index.html            # Frontend UI
├── tokenizer.pkl         # Saved tokenizer state
├── best_model.weights.h5 # Saved model weights
├── X.npy                 # Cached training sequences (input)
├── Y.npy                 # Cached training sequences (target)
└── loss_curve.png        # Training loss visualization
```

---

## Setup & Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (tested on NVIDIA RTX 3050 6GB)
- PostgreSQL 14

### Install Python Dependencies

```bash
python -m venv tfenv
source tfenv/bin/activate
pip install tensorflow==2.15.0
pip install fastapi uvicorn
pip install sqlalchemy psycopg2-binary
pip install datasets matplotlib numpy
```

### Install and Start PostgreSQL

```bash
sudo apt install postgresql postgresql-contrib
sudo service postgresql start

sudo -u postgres psql
CREATE DATABASE lil_gpt;
GRANT ALL PRIVILEGES ON DATABASE lil_gpt TO your_user;
\q
```

### Configure Database Connection

Update `database.py`:
```python
DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/lil_gpt"
```

### Run the Server

```bash
source tfenv/bin/activate
uvicorn server:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API UI.

---

## Training Guidelines

### From Scratch

```python
# in model.py, uncomment these lines:
tokenizer.train(text)
tokenizer.build_token_mappings(text)
tokenizer.save("tokenizer.pkl")
```

Then run:
```bash
python model.py
```

### Recommended Configuration by Hardware

| VRAM | batch_size | seq_len | num_layers | dff |
|------|-----------|---------|------------|-----|
| 4GB  | 16 | 128 | 6 | 1024 |
| 6GB  | 16 | 256 | 8 | 2048 |
| 8GB  | 32 | 256 | 8 | 2048 |
| 16GB | 64 | 512 | 12 | 3072 |

### Dataset Size vs Merges

| Stories | Recommended Merges |
|---------|--------------------|
| 10k | 1000 |
| 20k | 2000 |
| 50k | 3000 |
| 100k | 5000–6000 |

### Loss Targets

| Loss | Generation Quality |
|------|--------------------|
| > 3.0 | Random-looking output |
| 2.5–3.0 | Words and phrases but incoherent |
| 2.2–2.5 | Readable sentences, poor story coherence |
| 2.0–2.2 | Coherent paragraphs, some character drift |
| < 2.0 | Coherent stories with consistent characters |

### Tips

- Save `X.npy` and `Y.npy` after first encoding to avoid re-encoding on every run
- Use `patience=3, min_delta=0.01` in the callback — stops training automatically
- Monitor batch-level loss — spikes are normal, consistent upward trend is not
- Pre-LayerNorm is more stable than Post-LayerNorm for deep networks
- RoPE outperforms learned positional embeddings at longer sequence lengths

---

## Results

### Training History

| Run | Stories | Merges | seq_len | Layers | Final Loss |
|-----|---------|--------|---------|--------|------------|
| 1 | 5k | 500 | 64 | 6 | 2.97 |
| 2 | 20k | 4000 | 64 | 6 | 3.46 |
| 3 | 20k | 2000 | 64 | 6 | 2.74 |
| 4 | 50k | 3000 | 128 | 6 | 2.42 |
| 5 | 100k | 6000 | 256 | 8 | **2.32** |

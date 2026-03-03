# TinyGPT - Custom Transformer API

A small GPT-style transformer model trained on TinyStories and deployed using FastAPI.

## Features
- Custom Transformer architecture
- Rotary positional embeddings
- KV caching
- Top-k / Top-p / Temperature sampling
- PostgreSQL logging
- REST API with Swagger docs

## Download dataset from:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Model Specs
- d_model: 512
- Layers: 8
- Heads: 8
- dff: 2048
- Context length: 256
- Vocab size: ~6000



## Run Locally
pip install -r requirements.txt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from layers import TransformerBlock, PositionalEmbedding, create_causal_mask
from tokenizer import BPE_tokenizer
from model import GPT, generate_text, sample_top_p, sample_top_k, sample_with_temperature

if __name__ == "__main__":
    # load tokenizer
    tokenizer = BPE_tokenizer(num_merges=2000)
    tokenizer.load("tokenizer.pkl")

    seq_len = 256
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # build model with same architecture as training
    model = GPT(vocab_size=vocab_size,
                d_model=512,
                num_heads=8,
                dff=2048,
                num_layers=8,
                max_len=seq_len)

    # build model with dummy input before loading weights
    dummy = tf.constant(np.zeros((1, seq_len), dtype=np.int32))
    model(dummy, training=False)

    # load saved weights
    model.load_weights("best_model.weights.h5")
    print("Model weights loaded")

    # generate
    prompt = "Once upon a time there was a little girl"
    prompt_tokens = tokenizer.encode(prompt)
    start_tokens = tf.constant([prompt_tokens], dtype=tf.int32)

    print("\nGenerating...\n")
    output = generate_text(
        model,
        start_tokens,
        max_new_tokens=200,
        top_p=0.9,
        temperature=0.8
    )

    print(tokenizer.decode(output[0].numpy().tolist()))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from layers import TransformerBlock, PositionalEmbedding, create_causal_mask
from tokenizer import HFTokenizer
from model import GPT, generate_text, sample_top_p, sample_top_k, sample_with_temperature


if __name__ == "__main__":
    # load tokenizer (HuggingFace-backed, matches TinyStories training)
    tokenizer = HFTokenizer()
    tokenizer.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "saved_models", "tinystories_tokenizer.json",
    ))

    seq_len = 256
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # I match the training geometry exactly so weights load (~53M params)
    model = GPT(vocab_size=vocab_size,
                d_model=640,
                num_heads=10,
                dff=2560,
                num_layers=10,
                max_len=seq_len)

    # build model with dummy input before loading weights
    dummy = tf.constant(np.zeros((1, seq_len), dtype=np.int32))
    model(dummy, training=False)

    # load saved weights (training writes them to ../saved_models/)
    WEIGHTS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "saved_models", "tinystories_model.weights.h5",
    )
    model.load_weights(WEIGHTS_PATH)
    print("Model weights loaded")

    # generate — prompt can be passed on the command line, else use a default
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Once upon a time there was a little girl"
    prompt_tokens = tokenizer.encode(prompt)
    start_tokens = tf.constant([prompt_tokens], dtype=tf.int32)

    print("\nGenerating...\n")
    output = generate_text(
        model,
        start_tokens,
        max_new_tokens=200,
        top_p=0.85,
        temperature=0.5,
        eos_token_id=tokenizer.eos_id,
        repetition_penalty=1.3,
    )
    print(tokenizer.decode(output[0].numpy().tolist()))
from tensorflow.keras import mixed_precision
# mixed_precision policy is set in model.py to avoid setting twice.

import math
import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=None, training=False):
    """
    q,k,v shape: (batch, num_heads, seq_len, depth)
    mask shape:  (seq_len_q, seq_len_k) — 1 = mask out, 0 = keep
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = tf.cast(matmul_qk, tf.float32) / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (tf.cast(mask, tf.float32) * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # GPT-3 applies dropout to attention probabilities (post-softmax).
    if attn_dropout is not None:
        attention_weights = attn_dropout(attention_weights, training=training)

    # softmax ran in float32; cast back to v's dtype (float16 under mixed precision)
    # so the weighted sum over values doesn't raise a dtype mismatch.
    attention_weights = tf.cast(attention_weights, v.dtype)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


# MULTIHEAD ATTENTION
class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, max_len, rate=0.1):
        super().__init__()
        self.supports_masking = True

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.max_len = max_len

        # GPT-2/3 init: N(0, 0.02). Output projection gets scaled init for residual stability.
        std = 0.02
        out_std = 0.02 / math.sqrt(2 * num_layers)
        kinit = tf.keras.initializers.RandomNormal(stddev=std)
        oinit = tf.keras.initializers.RandomNormal(stddev=out_std)

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=kinit, bias_initializer="zeros")
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=kinit, bias_initializer="zeros")
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=kinit, bias_initializer="zeros")
        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=oinit, bias_initializer="zeros")

        self.attn_dropout = tf.keras.layers.Dropout(rate)

        # Precompute RoPE sin/cos for max_len once.
        angles = build_rope_angles(max_len, self.depth, offset=0)
        self._rope_sin = tf.constant(tf.sin(angles), dtype=tf.float32)  # (max_len, depth)
        self._rope_cos = tf.constant(tf.cos(angles), dtype=tf.float32)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))   # (batch, seq_len, num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])                          # (batch, num_heads, seq_len, depth)

    def call(self, v, k, q, mask=None, cache=None, training=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if cache is not None and cache.get("k") is not None:
            position_offset = tf.shape(cache["k"])[2]
        else:
            position_offset = 0

        q, k = apply_rope(q, k,
                          offset=position_offset,
                          sin_table=self._rope_sin,
                          cos_table=self._rope_cos)

        if cache is not None and cache.get("k") is not None:
            k = tf.concat([cache["k"], k], axis=2)
            v = tf.concat([cache["v"], v], axis=2)

        new_cache = {"k": k, "v": v}

        scaled_attention, _ = scaled_dot_product_attention(
            q, k, v, mask, attn_dropout=self.attn_dropout, training=training
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return self.dense(concat_attention), new_cache


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Token embedding only — position is handled by RoPE inside attention.
    Kept under this name for backward compatibility.
    """
    def __init__(self, vocab_size, d_model, max_len, rate=0.1):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        x = self.token_embedding(x)
        return self.dropout(x, training=training)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, num_layers, max_len, rate=0.1):
        super().__init__()
        self.supports_masking = True

        self.mha = MultiheadAttention(d_model, num_heads, num_layers, max_len, rate=rate)

        std = 0.02
        out_std = 0.02 / math.sqrt(2 * num_layers)
        self.dense1 = tf.keras.layers.Dense(
            dff, activation="gelu",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std),
            bias_initializer="zeros",
        )
        self.dense2 = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=out_std),
            bias_initializer="zeros",
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None, cache=None, training=False):
        n1 = self.layernorm1(x)
        attn_output, new_cache = self.mha(n1, n1, n1, mask=mask, cache=cache, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        n2 = self.layernorm2(out1)
        ffn_output = self.dense1(n2)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output, new_cache


def create_causal_mask(seq_len_q, seq_len_k=None):
    """
    Returns a (seq_len_q, seq_len_k) mask where 1 = mask out (future), 0 = keep.
    For cached generation: seq_len_q=1, seq_len_k=cache_len+1 — mask is all zeros.
    """
    if seq_len_k is None:
        seq_len_k = seq_len_q
    # Position i in the new q corresponds to absolute position (seq_len_k - seq_len_q + i).
    # It can attend to absolute positions <= its own.
    i = tf.range(seq_len_q)[:, None] + (seq_len_k - seq_len_q)
    j = tf.range(seq_len_k)[None, :]
    mask = tf.cast(j > i, tf.float32)
    return mask


def build_rope_angles(seq_len, head_dim, offset=0):
    position = tf.cast(tf.range(offset, offset + seq_len), dtype=tf.float32)   # absolute positions
    dim = tf.cast(tf.range(head_dim), dtype=tf.float32)
    theta = 10000.0 ** (-2 * (dim // 2) / tf.cast(head_dim, tf.float32))
    angles = tf.einsum("i,j->ij", position, theta)
    return tf.cast(angles, tf.float32)


def rotate_half(x):
    """
    Splits the last dim into even/odd pairs and rotates each pair 90 degrees.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    paired = tf.stack([-x2, x1], axis=-1)
    return tf.reshape(paired, tf.shape(x))


def apply_rope(q, k, offset=0, sin_table=None, cos_table=None):
    """
    q, k: (batch, num_heads, seq_len, head_dim)
    Uses precomputed sin/cos tables when provided.
    """
    seq_len = tf.shape(q)[2]
    head_dim = tf.shape(q)[3]

    if sin_table is None or cos_table is None:
        angles = build_rope_angles(seq_len, head_dim, offset=offset)
        sin = tf.sin(angles)
        cos = tf.cos(angles)
    else:
        sin = sin_table[offset:offset + seq_len]
        cos = cos_table[offset:offset + seq_len]

    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    # under mixed_float16 q/k are float16 but the sin/cos tables are float32;
    # I cast them to q's dtype so the multiply doesn't raise a dtype mismatch.
    sin = tf.cast(sin, q.dtype)
    cos = tf.cast(cos, q.dtype)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

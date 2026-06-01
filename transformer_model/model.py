from tensorflow.keras import mixed_precision
# bfloat16 on Ampere+ (RTX 3050): same exponent range as fp32, so NO loss scaling needed.
mixed_precision.set_global_policy('mixed_bfloat16')

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# longer HF download timeout + offline-tolerant retries reduce streaming dropouts
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

import tensorflow as tf 
import numpy as np
from layers import TransformerBlock,PositionalEmbedding,create_causal_mask
from tokenizer import CharTokenizer,BPE_tokenizer,HFTokenizer

class training_callback:
    def __init__(self,patience=3,min_delta=0.01,save_path="best_model.weights.h5"):
        self.patience = patience          # epochs to wait before early stopping
        self.min_delta = min_delta        # minimum improvement to count
        self.save_path = save_path
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.history = []
    def on_epoch_end(self,epoch,loss,model):
        self.history.append(loss)

        #saving the best model
        if loss<self.best_loss-self.min_delta:
            print(f"Loss improved{self.best_loss:.4f}->{loss:.4f},saving model")
            self.best_loss=loss
            model.save_weights(self.save_path)
            self.wait=0
        else:
            self.wait+=1
            print(f"No improvement for {self.wait}/{self.patience} epochs")

            #early stopping
            if self.wait>=self.patience:
                self.stopped_epoch=epoch
                print(f"Early stopping at epoch {epoch+1}, best loss: {self.best_loss:.4f}")
                return True #signal stops
        return False 
    def plot_history(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history,marker="o")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("loss_curve.png")
        print("Loss curve saved to png")
               

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,peak_lr,total_steps,warmup_steps):
        self.peak_lr=peak_lr
        self.total_steps=total_steps
        self.warmup_steps=warmup_steps
    
    def __call__(self,step):
        step=tf.cast(step,tf.float32)

        #linear warmup
        warmup_lr=self.peak_lr*(step/self.warmup_steps)

        #cosine decay
        cosine_lr=self.peak_lr*0.1*(
            1+tf.cos(np.pi * step/self.total_steps))/2
        
        return tf.where(step<self.warmup_steps,warmup_lr,cosine_lr)



class GPT(tf.keras.Model):
    """
    ORIGINAL ARCHITECTURE: POST-LN AND NO WEIGHT TYING
    GPT 1 USED WEIGHT TYING
    GPT2 USED POST-LAYERNORM (normalisation after attention)
    GPT3 USES PRE-LAYERNORM (normalisation before attention)
    """
    def __init__(self,vocab_size,d_model,num_heads,dff,num_layers,max_len,dropout_rate=0.1):
        super().__init__()
        self.num_layers=num_layers
        self.max_len=max_len

        self.embedding = PositionalEmbedding(vocab_size,d_model,max_len,rate=dropout_rate)
        #output shape:(batch,seq_len,d_model)

        self.blocks=[
            TransformerBlock(d_model,num_heads,dff,num_layers,max_len,rate=dropout_rate)
            for _ in range(num_layers)
        ]
        
        #creates as many transformer blocks as num_layers
        #output shape: (batch,seq_len,d_model)

        self.final_ln=tf.keras.layers.LayerNormalization()      #final normalization layer

       #self.final_layer=tf.keras.layers.Dense(vocab_size) REPLACED USING WEIGHT TYING
        #output shape: (d_model,vocab_shape)

    def call(self,x,training=False,caches=None):
        seq_len=tf.shape(x)[1]

        if caches is not None and len(caches)>0 and caches[0] is not None and caches[0].get("k") is not None:
            cache_len=tf.shape(caches[0]["k"])[2]
        else:
            cache_len=0

        seq_len_k=seq_len+cache_len
        mask=create_causal_mask(seq_len,seq_len_k)

        x=self.embedding(x,training=training)

        if caches is None:
            caches=[None for _ in self.blocks]

        new_caches=[]
        for block,cache in zip(self.blocks, caches):
            x,updated=block(x,mask=mask,cache=cache,training=training)
            new_caches.append(updated)

        x = self.final_ln(x)

        logits= tf.matmul(
            x,
            self.embedding.token_embedding.embeddings,
            transpose_b=True
        )
        logits=tf.cast(logits,tf.float32)

        return logits,new_caches

def sample_with_temperature(logits,temperature=1.0):

    logits=logits/temperature

    probabilities=tf.nn.softmax(logits)

    next_token=tf.random.categorical(
        tf.math.log(probabilities),num_samples=1
    )
    return tf.squeeze(next_token,axis=-1)  


def sample_top_k(logits,k=50):
    values,indices=tf.math.top_k(logits,k=k)

    min_values=values[:,-1,tf.newaxis]

    logits=tf.where(
        logits<min_values,
        tf.ones_like(logits)*-1e9,
        logits
        )
    probabilities=tf.nn.softmax(logits)

    next_token=tf.random.categorical(
        tf.math.log(probabilities),num_samples=1
    )

    return tf.squeeze(next_token,axis=-1)


def sample_top_p(logits,p=0.9):
    sorted_logits=tf.sort(logits,direction="DESCENDING")
    sorted_indices=tf.argsort(logits,direction="DESCENDING")

    cumulative_probs=tf.cumsum(tf.nn.softmax(sorted_logits),axis=-1)

    cutoff=cumulative_probs > p
    cutoff = tf.cast(cutoff,tf.int32)

    cutoff_index = tf.argmax(cutoff,axis=-1)

    threshold = tf.gather(
        sorted_logits,
        cutoff_index,
        batch_dims=1
    )

    logits =tf.where(
        logits<threshold[:,tf.newaxis],
        tf.ones_like(logits)*-1e9,
        logits)
    probabilities = tf.nn.softmax(logits)

    next_token=tf.random.categorical(
        tf.math.log(probabilities),num_samples=1
    )
    
    return tf.squeeze(next_token,axis=-1)



def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty is None or penalty == 1.0:
        return logits
    vocab_size = tf.shape(logits)[-1]
    ids = tf.reshape(generated_ids, [-1])
    counts = tf.math.bincount(ids, minlength=vocab_size, maxlength=vocab_size)
    seen = tf.cast(counts > 0, logits.dtype)[tf.newaxis, :]
    pos = logits > 0
    penalized = tf.where(pos, logits / penalty, logits * penalty)
    return logits * (1.0 - seen) + penalized * seen


def generate_text(model,
                  start_tokens,
                  max_new_tokens,
                  temperature=1.0,
                  top_k=None,
                  top_p=None,
                  eos_token_id=None,
                  repetition_penalty=1.0,
                  max_len=256):
    logits, caches = model(start_tokens, training=False)

    for _ in range(max_new_tokens):
        last_token_logits = logits[:, -1, :]
        last_token_logits = apply_repetition_penalty(last_token_logits, start_tokens, repetition_penalty)
        last_token_logits = last_token_logits / temperature

        if top_k is not None:
            next_token = sample_top_k(last_token_logits, k=top_k)
        elif top_p is not None:
            next_token = sample_top_p(last_token_logits, p=top_p)
        else:
            next_token = sample_with_temperature(last_token_logits, temperature=1.0)

        next_token = tf.cast(next_token, tf.int32)
        next_token = tf.expand_dims(next_token, axis=1)

        start_tokens = tf.concat([start_tokens, next_token], axis=1)

        if eos_token_id is not None and int(next_token[0, 0].numpy()) == int(eos_token_id):
            break

        logits, caches = model(next_token, training=False, caches=caches)

    return start_tokens

def create_sequences(encoded_text,seq_len):
    inputs=[]
    targets=[]

    for i in range(0,len(encoded_text) - seq_len, seq_len):
        input_seq=encoded_text[i:i+seq_len]
        target_seq=encoded_text[i+1:i+seq_len+1]

        inputs.append(input_seq)
        targets.append(target_seq)
    return np.array(inputs),np.array(targets)


def stream_documents(dataset_specs, max_docs=None, loop=False):
    """
    I yield raw text documents from a mix of streaming HF datasets.
    dataset_specs: list of (name, config, weight) — weight sets the interleave probability.
    I stop after max_docs documents (None = no limit).
    loop=True: when the dataset is exhausted I restart it from the beginning, so a
    token budget larger than the dataset can still be met (multiple passes).
    I retry on transient network errors instead of letting the whole run crash.
    """
    import time
    from datasets import load_dataset, interleave_datasets

    def build_mixed():
        streams, probs = [], []
        for name, config, weight in dataset_specs:
            ds = load_dataset(name, config, split="train", streaming=True)
            # I keep only the shared "text" column so datasets with different schemas
            # can be interleaved without column-mismatch errors.
            keep = [c for c in ds.column_names if c != "text"] if ds.column_names else []
            if keep:
                ds = ds.remove_columns(keep)
            streams.append(ds)
            probs.append(weight)
        return interleave_datasets(streams, probabilities=probs, seed=42)

    emitted = 0
    attempt = 0
    while True:
        try:
            mixed = build_mixed()
            it = iter(mixed)
            # I skip docs already emitted so a mid-stream rebuild doesn't repeat data.
            for _ in range(emitted):
                next(it, None)
            for item in it:
                if max_docs is not None and emitted >= max_docs:
                    return
                text = item.get("text")
                if text:
                    emitted += 1
                    yield text
            # stream exhausted: restart for another pass if looping, else stop.
            if loop:
                emitted = 0
                attempt = 0
                continue
            return
        except Exception as e:
            # transient network/DNS/HTTP errors: back off and rebuild the stream.
            attempt += 1
            wait = min(60, 2 ** attempt)
            print(f"[stream] error: {e!r} — retrying in {wait}s (attempt {attempt})", flush=True)
            time.sleep(wait)


def token_stream(doc_iterator, tokenizer, max_tokens=None):
    """
    I tokenize each document, append <|endoftext|> as a boundary, and yield token ids
    one at a time. I stop after max_tokens ids (None = no limit).
    """
    eos = tokenizer.eos_id
    produced = 0
    for doc in doc_iterator:
        ids = tokenizer.encode(doc)
        if eos is not None:
            ids = ids + [eos]                 # boundary so the model learns to stop
        for tid in ids:
            yield tid
            produced += 1
            if max_tokens is not None and produced >= max_tokens:
                return


def make_tf_dataset(token_iter_fn, seq_len, batch_size):
    """
    I turn a token-id generator into a batched (x, y) tf.data pipeline.
    Each sample: x = ids[i:i+seq_len], y = ids[i+1:i+seq_len+1]  -> next-token prediction.
    Shapes: x,y = (batch_size, seq_len).
    """
    def gen():
        buffer = []
        for tid in token_iter_fn():
            buffer.append(tid)
            # +1 because the target is shifted by one position
            if len(buffer) == seq_len + 1:
                x = np.array(buffer[:-1], dtype=np.int32)   # (seq_len,)
                y = np.array(buffer[1:],  dtype=np.int32)   # (seq_len,)
                yield x, y
                buffer = []                                  # non-overlapping windows

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    # 6GB card: I let TF grow VRAM incrementally instead of grabbing it all up front,
    # so I get real OOM signals instead of cryptic early crashes.
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # gradient accumulation: I sum grads over accum_steps micro-batches, then update once
    # effective batch = batch_size * accum_steps, but VRAM only holds one micro-batch
    accum_steps = 4

    @tf.function
    def micro_step(x, y):
        # x,y shape: (batch_size, seq_len)
        with tf.GradientTape() as tape:
            logits, _ = model(x, training=True)        # logits: (batch_size, seq_len, vocab_size)
            loss = loss_fn(y, logits)
            loss = loss / accum_steps                  # divide so summed grads average correctly

        gradients = tape.gradient(loss, model.trainable_variables)  # bf16 needs no loss scaling
        for buffer, grad in zip(accum_grads, gradients):    # I add into buffers, don't apply yet
            if grad is not None:
                buffer.assign_add(grad)
        return loss * accum_steps                      # I undo the divide so logs show real loss

    @tf.function
    def apply_accum():
        # I flush the summed grads as one update, then zero the buffers
        optimizer.apply_gradients(zip(accum_grads, model.trainable_variables))
        for buffer in accum_grads:
            buffer.assign(tf.zeros_like(buffer))
    
    # ---- tokenizer (HuggingFace-backed) ----
    # TinyStories has a small, uniform vocabulary, so a 10k vocab is plenty.
    seq_len = 256
    TOKENIZER_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "saved_models", "tinystories_tokenizer.json",
    )

    # TinyStoriesV2 (GPT-4) only: narrow, uniform, high-quality story domain that a
    # 50M model can actually master. ~2.75M stories ≈ 520-580M tokens; with loop=True
    # the 700M budget makes ~1.2 passes. noanabeshima/TinyStoriesV2 is streamable.
    DATASET_MIX = [
        ("noanabeshima/TinyStoriesV2", None, 1.0),
    ]

    tokenizer = HFTokenizer(vocab_size=10000)
    if os.path.exists(TOKENIZER_PATH):
        tokenizer.load(TOKENIZER_PATH)
    else:
        # I train the tokenizer on a bounded sample, then reuse it for the full run.
        print("Training HF tokenizer on a sample...")
        sample_docs = stream_documents(DATASET_MIX, max_docs=200_000)
        tokenizer.train(sample_docs)
        tokenizer.save(TOKENIZER_PATH)

    vocab_size = tokenizer.vocab_size
    print("Vocab_size: ", vocab_size)

    # ---- training budget (sized for ~2 days on a single 6GB GPU) ----
    # micro-batch = 4 fits a 53M model in 6GB; effective batch = 4 * 4 = 16
    batch_size = 4
    # TinyStoriesV2 is ~520-580M unique tokens; I train ~700M (≈1.2 passes via loop=True)
    # for the ~50M model, well within the 2-day budget on the 3050.
    MAX_TOKENS = 700_000_000

    # I count optimizer updates: tokens / (seq_len * effective_batch).
    effective_batch = batch_size * accum_steps
    total_steps = MAX_TOKENS // (seq_len * effective_batch)
    warmup_steps = total_steps // 10
    print("Planned optimizer updates:", total_steps)

    lr_schedule = WarmupCosineDecay(peak_lr=3e-4,
                                    total_steps=total_steps,
                                    warmup_steps=warmup_steps)

    optimizer = tf.keras.optimizers.Adam(lr_schedule,
                                         beta_1=0.9,
                                         beta_2=0.95,
                                         epsilon=1e-8,
                                         clipnorm=1.0)
    # no LossScaleOptimizer: mixed_bfloat16 doesn't underflow, so scaling is unnecessary.

    # ---- streaming data pipeline (no giant .npy, tokenized on the fly) ----
    # I hold out the first slice of tokens for validation, rest for training.
    VAL_TOKENS = 2_000_000

    def train_token_iter():
        # loop=True so the 700M budget can be met from the ~520-580M-token dataset (~1.2 passes).
        docs = stream_documents(DATASET_MIX, loop=True)
        # skip the validation tokens, then stream up to MAX_TOKENS for training
        stream = token_stream(docs, tokenizer, max_tokens=MAX_TOKENS + VAL_TOKENS)
        for idx, tid in enumerate(stream):
            if idx >= VAL_TOKENS:
                yield tid

    dataset = make_tf_dataset(train_token_iter, seq_len, batch_size)        # (batch, seq_len)

    # I materialize the validation set ONCE into memory (small, ~2M tokens) so that
    # periodic eval never touches the network. A transient outage during eval was what
    # crashed the earlier run, so the val set must be offline.
    print("Building in-memory validation set...")
    _val_ids = []
    for _tid in token_stream(stream_documents(DATASET_MIX), tokenizer, max_tokens=VAL_TOKENS):
        _val_ids.append(_tid)
    _n = (len(_val_ids) - 1) // seq_len
    _vx = np.array([_val_ids[k*seq_len:(k+1)*seq_len] for k in range(_n)], dtype=np.int32)
    _vy = np.array([_val_ids[k*seq_len+1:(k+1)*seq_len+1] for k in range(_n)], dtype=np.int32)
    val_dataset = (tf.data.Dataset.from_tensor_slices((_vx, _vy))
                   .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    print(f"Validation set: {len(_val_ids)} tokens -> {_n} sequences")

    model = GPT(vocab_size=vocab_size,
                d_model=640,              # embedding/hidden dim
                num_heads=10,             # head_dim = 640/10 = 64
                dff=2560,                 # FFN inner dim = 4 * d_model
                num_layers=10,            # 10 stacked blocks, ~53M params total
                max_len=seq_len)

    # I build once so trainable_variables exist before I make the grad buffers
    model(tf.zeros((1, seq_len), dtype=tf.int32), training=False)   # input: (1, seq_len)

    # one zero buffer per weight, same shape as each variable; grads accumulate here
    accum_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False)
        for v in model.trainable_variables
    ]

    # I build the optimizer's slot variables now so a checkpoint restore can populate
    # Adam moments + the step counter before training starts.
    optimizer.build(model.trainable_variables)

    @tf.function
    def val_step(x, y):
        # I evaluate without dropout/grads to get a generalization signal.
        logits, _ = model(x, training=False)
        return loss_fn(y, logits)

    def evaluate(max_batches=200):
        losses = []
        for j, (vx, vy) in enumerate(val_dataset):
            if j >= max_batches:
                break
            losses.append(val_step(vx, vy).numpy())
        return float(np.mean(losses)) if losses else float("inf")

    SAVE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "saved_models", "tinystories_model.weights.h5",
    )
    EVAL_EVERY = 1000        # optimizer updates between validation checks

    # ---- resumable checkpoint: model + optimizer + progress counter ----
    # I track update_step in a tf.Variable so it's saved/restored with the checkpoint;
    # a 2-day run can then survive a crash and pick up where it stopped.
    # NOTE: fresh dir name — the 10k-vocab TinyStories model must not load the old
    # 16k-vocab general checkpoint (shapes differ, would crash on restore).
    ckpt_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_val_var = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)
    CKPT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "saved_models", "ckpt_tinystories",
    )
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer,
                               step=ckpt_step, best_val=best_val_var)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=2)

    start_step = 0
    if ckpt_mgr.latest_checkpoint:
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
        start_step = int(ckpt_step.numpy())
        best_val = float(best_val_var.numpy())
        print(f"Resumed from {ckpt_mgr.latest_checkpoint} at update {start_step}, best_val {best_val:.4f}")
    else:
        best_val = float("inf")
        print("No checkpoint found, starting fresh.")

    # ---- single-pass training loop with step-based checkpointing ----
    # I run one pass over ~600M tokens; epochs don't apply to a streaming run.
    # On resume I fast-forward the stream past tokens already consumed.
    update_step = start_step
    micro_count = 0
    skip_micro = start_step * accum_steps     # micro-batches already trained before the crash

    for i, (batch_x, batch_y) in enumerate(dataset):
        # I skip batches already seen in a previous run so I don't retrain on them.
        if i < skip_micro:
            continue

        # I accumulate every batch, apply once every accum_steps
        loss = micro_step(batch_x, batch_y)
        micro_count += 1

        if micro_count % accum_steps == 0:
            apply_accum()
            update_step += 1
            ckpt_step.assign(update_step)

            if update_step % 50 == 0:
                print(f"update {update_step}/{total_steps}, loss: {loss.numpy():.4f}")

            # I checkpoint on validation loss, not training loss, to avoid saving an overfit model.
            if update_step % EVAL_EVERY == 0:
                val_loss = evaluate()
                print(f"  [eval] update {update_step}, val_loss: {val_loss:.4f}")
                if val_loss < best_val:
                    print(f"  val improved {best_val:.4f} -> {val_loss:.4f}, saving")
                    best_val = val_loss
                    best_val_var.assign(best_val)
                    model.save_weights(SAVE_PATH)          # best weights (for inference)
                # I always snapshot the resumable checkpoint (model+optimizer+step).
                ckpt_mgr.save()

        if update_step >= total_steps:
            break

    print("Training complete. best val_loss:", best_val)

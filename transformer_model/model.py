from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from layers import TransformerBlock,PositionalEmbedding,create_causal_mask
from tokenizer import CharTokenizer,BPE_tokenizer
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

        self.embedding = PositionalEmbedding(vocab_size,d_model,max_len)
        #output shape:(batch,seq_len,d_model)

        self.blocks=[
            TransformerBlock(d_model,num_heads,dff,dropout_rate)
            for _ in range(num_layers)
        ]
        
        #creates as many transformer blocks as num_layers
        #output shape: (batch,seq_len,d_model)

        self.final_ln=tf.keras.layers.LayerNormalization()      #final normalization layer

       #self.final_layer=tf.keras.layers.Dense(vocab_size) REPLACED USING WEIGHT TYING
        #output shape: (d_model,vocab_shape)

    def call(self,x,training =False,caches=None):
        seq_len=tf.shape(x)[1]
        mask=create_causal_mask(seq_len)
        x = self.embedding(x)

        if caches is None:
            caches=[{} for _ in self.blocks]    #one caceh dictionary per layer


        for block,cache in zip(self.blocks, caches):
            x=block(x,mask=mask,cache=cache,training=training)
        
        x = self.final_ln(x)    

        logits= tf.matmul(
            x,
            self.embedding.token_embedding.embeddings, 
            #embeddings is the actual learned weight matrix of this layer
            transpose_b=True
        )
        logits=tf.cast(logits,tf.float32)

        return logits,caches

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


def sample_top_p(logits,p=0.09):
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



def generate_text(model,
                  start_tokens,max_new_tokens,
                  temperature=1.0,
                  top_k=None,
                  top_p=None):
    logits,caches=model(start_tokens,training=False)
    
    for i in range(max_new_tokens):
        last_token_logits = logits[:,-1,:]
        last_token_logits=last_token_logits/temperature

        if top_k is not None:
            next_token=sample_top_k(last_token_logits,k=top_k)
        elif top_p is not None:
            next_token=sample_top_p(last_token_logits,p=top_p)
        
        else:
            next_token=sample_with_temperature(last_token_logits,temperature=temperature)
        
        next_token=tf.cast(next_token,tf.int32)
        next_token = tf.expand_dims(next_token,axis=1)

        logits,caches=model(next_token,training=False,caches=caches)

        start_tokens=tf.concat([start_tokens,next_token],axis=1)
    
    return start_tokens

def create_sequences(encoded_text,seq_len):
    inputs=[]
    targets=[]

    for i in range(1,len(encoded_text)-seq_len,seq_len):
        input_seq=encoded_text[i:i+seq_len]
        target_seq=encoded_text[i+1:i+seq_len+1]

        inputs.append(input_seq)
        targets.append(target_seq)
    return np.array(inputs),np.array(targets)



    

if __name__ == "__main__":
    from datasets import load_dataset

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_steps(x,y):
        with tf.GradientTape() as tape:
            logits,_=model(x,training=True)
            loss =loss_fn(y,logits)
    
        gradients=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss 
    
    hf_dataset=load_dataset("roneneldan/TinyStories")    
    text = " ".join(hf_dataset["train"]["text"][:100000])

    tokenizer=BPE_tokenizer(num_merges=6000)    
    tokenizer.load("/home/avipandey/Projects/TinyGPT/saved_models/tokenizer.pkl")
    # tokenizer.train(text)    
    # tokenizer.build_token_mappings(text)
    # tokenizer.save("tokenizer.pkl")

    vocab_size=tokenizer.vocab_size
    print("Vocab_size: ",vocab_size)

    seq_len=256


    print("Encoding and creating sequences...")
    encoded_text=tokenizer.encode(text)
    X,Y=create_sequences(encoded_text,seq_len)
    np.save("X.npy",X)
    np.save("Y.npy",Y)
    print("Sequeces saved")
    
    print("input_shape: ",X.shape)
    print("target_shape: ",Y.shape)

    # we get (num_samples,128)  

    batch_size=16
    epochs=15

    total_steps=(len(X)//batch_size)* epochs
    warmup_steps=total_steps//10
    lr_schedule=WarmupCosineDecay(peak_lr=3e-4,
                                   total_steps=total_steps,
                                   warmup_steps=warmup_steps)
    
    optimizer = tf.keras.optimizers.Adam(lr_schedule,
                                      beta_1=0.9,
                                      beta_2=0.95,
                                      epsilon=1e-8,
                                      clipnorm=1.0)
    
    dataset=tf.data.Dataset.from_tensor_slices((X,Y))
    dataset=dataset.shuffle(10000).batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # we get (batch_size,seq_len)
    
    vocab_size=tokenizer.vocab_size

    model=GPT(vocab_size=vocab_size,
              d_model=512,              #input dimensions
              num_heads=8,              #number of attention heads
              dff=2048,                 #dimension of feed forward network
              num_layers=8,             #6 transformer layers
              max_len=seq_len)   
   
    #TRAINING LOOP

    callback=training_callback(patience=3,min_delta=0.01)
    
    for epoch in range(epochs):
         epoch_losses=[]
         for i,(batch_x,batch_y) in enumerate(dataset):
             loss=train_steps(batch_x,batch_y)
             epoch_losses.append(loss.numpy())
             if i%50==0:
                 print(f"batch{i},loss:{loss.numpy():.4f}")
         epoch_loss=np.mean(epoch_losses)
         print(f"Epoch{epoch+1},loss:{epoch_loss:.4f}")
         stop=callback.on_epoch_end(epoch,epoch_loss,model)
         if stop:
             break
    callback.plot_history()
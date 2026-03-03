from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

import tensorflow as tf
import numpy as np 

def scaled_dot_product_attention(q,k,v,mask=None):
    matmul_qk=tf.matmul(q,k,transpose_b=True)

#q_shape = k_shape = (batch, seq_len_q, depth)
#malmut_qk_shape = (batch, seq_len_q, seq_len_k)
    
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits=tf.cast(matmul_qk,tf.float32)/tf.math.sqrt(dk)
    #masking so that the decoder does not look at future tokens 
    if mask is not None:
        scaled_attention_logits +=(tf.cast(mask,tf.float32)*-1e9)
    #applying softmax to get the attention weights
    attention_weights=tf.nn.softmax(scaled_attention_logits,axis=-1)

    #multiply with v 
    output = tf.matmul(attention_weights,v)

    return (output,attention_weights)
#this give s us softmax(QKt/dk^-2)V

#MULTIHEAD ATTENTION
class MultiheadAttention(tf.keras.layers.Layer): 
    def __init__(self,d_model,num_heads):
        super().__init__()

        self.supports_masking= True 
        
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model%num_heads==0

        self.depth = d_model//num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model) 

        self.dense = tf.keras.layers.Dense(d_model)

        #gives (batch,seq_len,d_model)

    def split_heads(self,x,batch_size):
        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))   #turns input to (batch,seq_len,num_heads,depth)
        return tf.transpose(x,perm=[0,2,1,3])                       #turns it to (batch,num_heads,seq_len,depth)
    
    def call(self,v,k,q,mask=None,cache=None):
        batch_size=tf.shape(q)[0]   #gets the batch size dynamically
        q=self.wq(q)
        k=self.wk(k)
        v=self.wv(v)

        q=self.split_heads(q,batch_size)        #splits heads , giving each head its own space q,k and v
        k=self.split_heads(k,batch_size)
        v=self.split_heads(v,batch_size)

        q,k=apply_rope(q,k)

        if cache is not None:
            if "k" in cache:
                k=tf.concat([cache["k"],k],axis=2)
                v=tf.concat([cache["v"],v],axis=2)
            cache["k"] = k
            cache["v"] = v


        scaled_attention,attention_weights= scaled_dot_product_attention(q,k,v,mask)    #applies attention to all heads separately

        #process of combining all the heads again 
        scaled_attention=tf.transpose(scaled_attention,perm=[0,2,1,3])  #transposing back to the previous config          

        concat_attention=tf.reshape(                                    #combining back to the previous config (batch,seq_len,d_model)
            scaled_attention,
            (batch_size,-1,self.d_model)
            )
        
        output = self.dense(concat_attention)

        return output

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model,max_len):
        super().__init__()

        self.token_embedding=tf.keras.layers.Embedding(vocab_size,d_model)
        

    def call(self,x):
        return self.token_embedding(x)

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super().__init__()

        self.supports_masking=True 

        self.mha =MultiheadAttention(d_model,num_heads)

        self.dense1 = tf.keras.layers.Dense(dff,activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model)

        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1=tf.keras.layers.Dropout(rate)
        self.dropout2=tf.keras.layers.Dropout(rate)

    def call(self,x,mask=None,cache=None,training=False):
        seq_len=tf.shape(x)[1]
        mask=create_causal_mask(seq_len)        #masking

        attn_output=self.mha(self.layernorm1(x),self.layernorm1(x),self.layernorm1(x),mask,cache=cache)
        attn_output=self.dropout1(attn_output,training=training)

        out1=x+attn_output                      #residual after attention

        ffn_output=self.dense1(self.layernorm2(out1))   #pre-layernorm
        ffn_output=self.dense2(ffn_output)
        ffn_output=self.dropout2(ffn_output,training=training)

        return out1 + ffn_output                       #no LN as end 

def create_causal_mask(seq_len):
    mask = 1-tf.linalg.band_part(                   #keeps the lower triangle only bcos we need to mask the previous tokens only 
        tf.ones((seq_len,seq_len)),-1,0
    )
    return mask
   
def build_rope_angles(seq_len,head_dim):


    position=tf.cast(tf.range(seq_len),dtype=tf.float32)     #absolute positional index
    #shape = (seq_len,)

    dim =tf.cast(tf.range(head_dim),dtype=tf.float32)
    #shape=(head_dim,)      
    theta = 10000**(-2*(dim//2)/tf.cast(head_dim,tf.float32)) #calculates frequency for each dimension
    #gives low frequency for later dims and high for early dims
    angles=tf.einsum("i,j->ij",position,theta)  

    return tf.cast(angles,tf.float32)   #shape=(seq_len,head_dim)       [m,d] how much position m is rotated for dimension d

def rotate_half(x):
    """
    splits vectors into even and odd 
    dimensions,forming 2d pairs that rotate together 
    """
    x1=x[...,::2]
    x2=x[...,1::2]    
    paired=tf.stack([-x2,x1],axis=-1)
    return tf.reshape(paired,tf.shape(x))   #makes the rotated partner of each pair 

def apply_rope(q,k):
    seq_len=tf.shape(q)[2]
    head_dim=tf.shape(q)[3]

    #compute sin and cos of all angles
    angles=build_rope_angles(seq_len,head_dim)
    sin=tf.sin(angles)
    cos=tf.cos(angles)
    #convert to shape (batch,num_heads,seq_len,head_dim)
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]    


    q_rot=q*cos + rotate_half(q)*sin    
    k_rot=k*cos + rotate_half(k)*sin

    return q_rot,k_rot


    




    




    
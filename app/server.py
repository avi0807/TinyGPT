#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
import sys, os

# path fixes
sys.path.append(os.path.join(os.path.dirname(__file__)))               # app/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'transformer_model'))  # transformer_model/

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, get_db
import crud as crud

SEQ_LEN   = 256
WEIGHTS   = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'best_model.weights.h5')
TOKENIZER = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'tokenizer.pkl')
INDEX_HTML = os.path.join(os.path.dirname(__file__), '..', 'index.html')

def get_model():
    if not hasattr(app.state, "model"):
        print("Loading model for the first time...")

        import tensorflow as tf
        from model import GPT, generate_text
        from tokenizer import BPE_tokenizer

        tokenizer = BPE_tokenizer(num_merges=6000)
        tokenizer.load(TOKENIZER)

        model = GPT(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            num_heads=8,
            dff=2048,
            num_layers=8,
            max_len=SEQ_LEN,
        )

        dummy = tf.zeros((1, SEQ_LEN), dtype=tf.int32)
        model(dummy, training=False)
        model.load_weights(WEIGHTS)

        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.generate_text = generate_text

        print("Model loaded.")

    return app.state.model, app.state.tokenizer, app.state.generate_text


app = FastAPI(
    title="TinyStories GPT",
    description="A small GPT model trained on TinyStories",
    version="1.0.0",
)
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(default=100, ge=10, le=300)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    temperature: float
    top_p: float
    max_new_tokens: int
    response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    vocab_size: int
    model_params: str
    seq_len: int


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(INDEX_HTML)


@app.get("/health", response_model=HealthResponse)
def health():
    if not hasattr(app.state, "model"):
        return HealthResponse(
            status="model_not_loaded",
            vocab_size=0,
            model_params="0M",
            seq_len=SEQ_LEN,
        )

    import tensorflow as tf
    total_params = sum(
        tf.size(w).numpy() for w in app.state.model.trainable_variables
    )

    return HealthResponse(
        status="healthy",
        vocab_size=app.state.tokenizer.vocab_size,
        model_params=f"{total_params / 1e6:.1f}M",
        seq_len=SEQ_LEN,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, db: Session = Depends(get_db)):
    model, tokenizer, generate_text = get_model()
    import tensorflow as tf
    try:
        start_time = time.time()

        prompt_tokens = tokenizer.encode(request.prompt)
        if len(prompt_tokens) > SEQ_LEN - 10:
            prompt_tokens = prompt_tokens[-(SEQ_LEN - 10):]

        start_tokens = tf.constant([prompt_tokens], dtype=tf.int32)

        output = generate_text(
            model,
            start_tokens,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            temperature=request.temperature,
        )

        output_ids = output[0].numpy().tolist()
        generated = tokenizer.decode(output_ids)
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        crud.save_generation(
            db=db,
            prompt=request.prompt,
            generated_text=generated,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            response_time_ms=response_time_ms,
        )

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            response_time_ms=response_time_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    records = crud.get_all_generations(db, limit=limit)
    return [
        {
            "id": r.id,
            "prompt": r.prompt,
            "generated_text": r.generated_text,
            "temperature": r.temperature,
            "top_p": r.top_p,
            "max_new_tokens": r.max_new_tokens,
            "response_time_ms": r.response_time_ms,
            "created_at": r.created_at,
        }
        for r in records
    ]


@app.delete("/history/{generation_id}")
def delete_history_entry(generation_id: int, db: Session = Depends(get_db)):
    deleted = crud.delete_generation(db, generation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"deleted": generation_id}
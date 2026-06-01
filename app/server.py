import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'transformer_model'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set CUDA_VISIBLE_DEVICES=-1 in your environment to force CPU inference.

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, get_db
import crud as crud

SEQ_LEN = 256
WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'tinystories_model.weights.h5')
TOKENIZER = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'tinystories_tokenizer.json')
INDEX_HTML = os.path.join(os.path.dirname(__file__), '..', 'index.html')

# Temperature is the single creativity slider. Low (left) = safe/coherent, high (right) = wild.
TEMP_MIN = 0.1
TEMP_MAX = 1.5
TEMP_STEP = 0.1
TEMP_DEFAULT = 0.5
REPETITION_PENALTY = 1.3

# Descriptive zones along the slider, low -> high. Each carries the colour the slider
# should show in that region so the track runs green (safe) to red (wild).
TEMPERATURE_ZONES = [
    {"from": 0.1, "to": 0.5, "label": "Safe",
     "color": "#2ecc71",
     "description": "Focused and predictable. The model picks the most likely words, so stories stay simple, calm and easy to follow."},
    {"from": 0.5, "to": 0.8, "label": "Balanced",
     "color": "#f1c40f",
     "description": "A good mix of sense and surprise. Stories stay coherent but take a few fun turns. Recommended for most prompts."},
    {"from": 0.8, "to": 1.1, "label": "Creative",
     "color": "#e67e22",
     "description": "More imaginative. The model reaches for less obvious words and ideas, so stories get more varied and playful."},
    {"from": 1.1, "to": 1.5, "label": "Wild",
     "color": "#e74c3c",
     "description": "Unpredictable and quirky. Anything can happen, but the story may wander or stop making sense."},
]

# Gradient stops used to compute the exact colour for any temperature: green -> yellow -> red.
_GRADIENT = [(0.0, (46, 204, 113)), (0.5, (241, 196, 15)), (1.0, (231, 76, 60))]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def temperature_color(temp: float) -> str:
    """Interpolate green -> yellow -> red across the temperature range, return hex."""
    pos = _clamp((temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0.0, 1.0)
    for i in range(len(_GRADIENT) - 1):
        p0, c0 = _GRADIENT[i]
        p1, c1 = _GRADIENT[i + 1]
        if p0 <= pos <= p1:
            t = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            r = round(c0[0] + (c1[0] - c0[0]) * t)
            g = round(c0[1] + (c1[1] - c0[1]) * t)
            b = round(c0[2] + (c1[2] - c0[2]) * t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#e74c3c"


def describe_temperature(temp: float):
    """Return (label, description, color) for a temperature value."""
    for z in TEMPERATURE_ZONES:
        if z["from"] <= temp < z["to"]:
            return z["label"], z["description"], temperature_color(temp)
    # temp == TEMP_MAX falls into the last zone
    last = TEMPERATURE_ZONES[-1]
    return last["label"], last["description"], temperature_color(temp)


def derive_top_p(temp: float) -> float:
    """One slider controls everything: scale top_p with temperature (0.80 -> 0.95)."""
    pos = _clamp((temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0.0, 1.0)
    return round(0.80 + pos * 0.15, 3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    import tensorflow as tf
    from model import GPT, generate_text
    from tokenizer import HFTokenizer

    tokenizer = HFTokenizer()
    tokenizer.load(TOKENIZER)

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=640,
        num_heads=10,
        dff=2560,
        num_layers=10,
        max_len=SEQ_LEN,
    )
    model(tf.zeros((1, SEQ_LEN), dtype=tf.int32), training=False)
    model.load_weights(WEIGHTS)

    app.state.tf = tf
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.generate_text = generate_text
    print("Model loaded.")
    yield


app = FastAPI(
    title="TinyStories GPT",
    description="A small GPT language model that writes short children's stories from a prompt.",
    version="2.0.0",
    lifespan=lifespan,
)
# DB is optional: if it's unreachable (e.g. ephemeral hosting without Postgres),
# the app still serves generations, it just won't persist history.
try:
    Base.metadata.create_all(bind=engine)
    DB_AVAILABLE = True
except Exception as e:
    print(f"DB unavailable, history disabled: {e}")
    DB_AVAILABLE = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500,
                        description="How the story should start.")
    temperature: float = Field(
        default=TEMP_DEFAULT, ge=TEMP_MIN, le=TEMP_MAX,
        description="Creativity slider. Low (green) = safe and coherent, high (red) = wild and random.",
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.1, le=1.0,
        description="Advanced override for nucleus sampling. Auto-derived from temperature if omitted.",
    )
    max_new_tokens: int = Field(default=200, ge=10, le=300,
                                description="Roughly how long the story can get.")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    temperature: float
    temperature_label: str
    temperature_description: str
    temperature_color: str
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


@app.get("/temperature-info")
def temperature_info():
    """Slider configuration + colour zones so the frontend can render a green->red track."""
    return {
        "min": TEMP_MIN,
        "max": TEMP_MAX,
        "step": TEMP_STEP,
        "default": TEMP_DEFAULT,
        "gradient": ["#2ecc71", "#f1c40f", "#e74c3c"],
        "zones": TEMPERATURE_ZONES,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    import tensorflow as tf
    total_params = sum(tf.size(w).numpy() for w in app.state.model.trainable_variables)
    return HealthResponse(
        status="healthy",
        vocab_size=app.state.tokenizer.vocab_size,
        model_params=f"{total_params / 1e6:.1f}M",
        seq_len=SEQ_LEN,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, db: Session = Depends(get_db)):
    tf = app.state.tf
    model = app.state.model
    tokenizer = app.state.tokenizer
    generate_text = app.state.generate_text

    temperature = request.temperature
    top_p = request.top_p if request.top_p is not None else derive_top_p(temperature)
    label, description, color = describe_temperature(temperature)

    try:
        start_time = time.time()

        prompt_tokens = tokenizer.encode(request.prompt)
        if len(prompt_tokens) > SEQ_LEN - 10:
            prompt_tokens = prompt_tokens[-(SEQ_LEN - 10):]

        start_tokens = tf.constant([prompt_tokens], dtype=tf.int32)

        output = await run_in_threadpool(
            generate_text,
            model,
            start_tokens,
            request.max_new_tokens,
            temperature,
            None,            # top_k
            top_p,
            getattr(tokenizer, "eos_id", None),
            REPETITION_PENALTY,
        )

        generated = tokenizer.decode(output[0].numpy().tolist())
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        if DB_AVAILABLE:
            try:
                crud.save_generation(
                    db=db,
                    prompt=request.prompt,
                    generated_text=generated,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=request.max_new_tokens,
                    response_time_ms=response_time_ms,
                )
            except Exception as e:
                print(f"save_generation failed (continuing): {e}")

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            temperature=temperature,
            temperature_label=label,
            temperature_description=description,
            temperature_color=color,
            top_p=top_p,
            max_new_tokens=request.max_new_tokens,
            response_time_ms=response_time_ms,
        )

    except HTTPException:
        raise
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

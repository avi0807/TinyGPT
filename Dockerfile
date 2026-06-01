FROM python:3.10-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3 \
    CUDA_VISIBLE_DEVICES=-1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf

# Slim, serving-only dependencies (CPU TensorFlow, no training/CUDA packages).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app/ ./app/
COPY transformer_model/ ./transformer_model/
COPY index.html .

# Only the artifacts the server needs (not the training checkpoints).
COPY saved_models/tinystories_tokenizer.json ./saved_models/
COPY saved_models/tinystories_model.weights.h5 ./saved_models/

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860}"]

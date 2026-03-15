FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY transformer_model/ ./transformer_model/
COPY saved_models/ ./saved_models/
COPY index.html .

CMD ["uvicorn", "app.server:app","--host", "0.0.0.0", "--port", "8000"]


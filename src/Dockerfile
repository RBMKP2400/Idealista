# ---------- STAGE 1: Builder ----------
FROM python:3.12-slim

WORKDIR /app

# Instalar "es_core_news_sm"
RUN pip install --no-cache-dir spacy \
    && python -m spacy download es_core_news_sm

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers

# Copiar requirements y código para instalar dependencias adicionales
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY connection/ ./connection/
COPY params/ ./params/
COPY pipelines/ ./pipelines/
COPY utils/ ./utils/
COPY config.yaml .
COPY main.py .

# Comando por defecto
CMD ["python", "-u", "main.py"]

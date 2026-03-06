FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Pre-download the model at build time so it's baked into the image
RUN python -c "from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation; \
    SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes'); \
    AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

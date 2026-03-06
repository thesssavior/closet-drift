FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Pre-download and export ONNX model at build time
RUN python -c "from transformers import SegformerImageProcessor; \
    from optimum.onnxruntime import ORTModelForSemanticSegmentation; \
    SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes'); \
    ORTModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes', export=True)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

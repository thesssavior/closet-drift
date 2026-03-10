FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY static/ static/

# Pre-download DETR + YOLOS + SAM2 + CLIP models at build time
RUN python -c "from transformers import AutoImageProcessor, AutoModelForObjectDetection, Sam2Processor, Sam2Model, CLIPModel, CLIPProcessor; \
    AutoImageProcessor.from_pretrained('facebook/detr-resnet-50'); \
    AutoModelForObjectDetection.from_pretrained('facebook/detr-resnet-50'); \
    AutoImageProcessor.from_pretrained('valentinafeve/yolos-fashionpedia'); \
    AutoModelForObjectDetection.from_pretrained('valentinafeve/yolos-fashionpedia'); \
    Sam2Processor.from_pretrained('facebook/sam2.1-hiera-tiny'); \
    Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

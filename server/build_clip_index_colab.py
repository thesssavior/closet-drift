"""
CLIP indexing script for Google Colab.

Instructions:
1. Open Google Colab, select GPU runtime (Runtime → Change runtime type → T4 GPU)
2. Upload this script, items_cache_s3.json
3. Run the cells below
"""

# === CELL 1: Install dependencies ===
# !pip install transformers torch pillow requests python-dotenv

# === CELL 2: Upload items_cache and set env vars ===
# from google.colab import files
# uploaded = files.upload()  # upload items_cache_s3.json

# === CELL 3: Set your credentials (paste these) ===
QDRANT_URL = "https://9229483c-751f-450a-baea-90b74292027b.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oOJEWHTl5Po66RMlDi6cb32XAJ23n2VBD2TyB1BN-f0"

# === CELL 4: Run indexing ===

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

COLLECTION = "fashionpedia_clip"
EMBED_DIM = 768  # ViT-L/14
CLIP_BATCH = 256
UPSERT_BATCH = 500
DOWNLOAD_WORKERS = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Load CLIP
print("Loading CLIP ViT-L/14...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
clip_model.to(DEVICE)
clip_model.eval()
print("CLIP loaded.")


def embed_images(images: list) -> list:
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().tolist()


def download_crop(url: str):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def download_batch(urls: list) -> list:
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        return list(pool.map(download_crop, urls))


def ensure_collection():
    r = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY}, timeout=10,
    )
    if r.status_code == 200:
        info = r.json().get("result", {})
        print(f"Collection '{COLLECTION}' exists, {info.get('points_count', '?')} points")
        return
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}, "optimizers_config": {"indexing_threshold": 0}},
        timeout=10,
    )
    resp.raise_for_status()
    print(f"Created collection '{COLLECTION}'")


def upsert_batch(points: list):
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"points": points}, timeout=60,
    )
    resp.raise_for_status()


def get_points_count() -> int:
    """Get current point count to figure out where to resume."""
    try:
        r = requests.get(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            headers={"api-key": QDRANT_API_KEY}, timeout=10,
        )
        return r.json().get("result", {}).get("points_count", 0)
    except Exception:
        return 0


# Load items
print("Loading items...")
with open("items_cache_s3.json") as f:
    items = json.load(f)
print(f"Loaded {len(items)} items")

ensure_collection()

# Resume from existing progress
existing = get_points_count()
start_from = existing
total = len(items)

if start_from >= total:
    print(f"All {total} items already indexed!")
else:
    if start_from > 0:
        print(f"Resuming: {start_from} items already in Qdrant, {total - start_from} remaining")

    remaining = items[start_from:]
    done = start_from
    t0 = time.time()
    download_fails = 0

    chunks = [remaining[i : i + UPSERT_BATCH] for i in range(0, len(remaining), UPSERT_BATCH)]

    # Prefetch pipeline
    prefetch_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)

    def start_prefetch(chunk):
        urls = [item["crop_url"] for item in chunk]
        return prefetch_pool.submit(download_batch, urls)

    prefetch_future = start_prefetch(chunks[0]) if chunks else None

    for ci, chunk in enumerate(chunks):
        images = prefetch_future.result()

        if ci + 1 < len(chunks):
            prefetch_future = start_prefetch(chunks[ci + 1])

        valid = [(item, img) for item, img in zip(chunk, images) if img is not None]
        download_fails += len(chunk) - len(valid)

        if not valid:
            done += len(chunk)
            continue

        all_vectors = []
        valid_items = [v[0] for v in valid]
        valid_images = [v[1] for v in valid]

        for i in range(0, len(valid_images), CLIP_BATCH):
            batch_imgs = valid_images[i : i + CLIP_BATCH]
            try:
                vecs = embed_images(batch_imgs)
                all_vectors.extend(vecs)
            except Exception as e:
                print(f"  [embed] error: {e}")
                all_vectors.extend([[0.0] * EMBED_DIM] * len(batch_imgs))

        points = []
        for item, vector in zip(valid_items, all_vectors):
            points.append({
                "id": item["id"],
                "vector": vector,
                "payload": {
                    "description": item["description"],
                    "category": item["category"],
                    "category_id": item["category_id"],
                    "attributes": item["attributes"],
                    "crop_url": item["crop_url"],
                    "width": item["width"],
                    "height": item["height"],
                },
            })

        try:
            upsert_batch(points)
        except Exception as e:
            print(f"Upsert error at item {done}: {e}")
            break

        done += len(chunk)

        elapsed = time.time() - t0
        rate = (done - start_from) / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] | {download_fails} dl fails | {rate:.0f} items/s | ETA {eta/60:.0f}m")

    prefetch_pool.shutdown()
    print(f"\nDone! {done}/{total} indexed in {(time.time()-t0)/60:.1f}m | {download_fails} download failures")

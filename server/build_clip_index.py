"""
Build CLIP image vector index from existing Fashionpedia crops.

Reads items_cache_s3.json (from the previous Gemini indexing run),
downloads crops from Supabase, embeds with CLIP ViT-L/14, upserts to Qdrant.

Usage:
  python -u build_clip_index.py
  python -u build_clip_index.py --fresh   # start over
"""

import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

QDRANT_URL = os.environ["QDRANT_CLIP_URL"]
QDRANT_API_KEY = os.environ["QDRANT_CLIP_API_KEY"]

COLLECTION = "fashionpedia_clip"
EMBED_DIM = 768  # CLIP ViT-L/14
CLIP_BATCH = 128
UPSERT_BATCH = 500
CHECKPOINT_EVERY = 500
DOWNLOAD_WORKERS = 64

# Use MPS (Apple Silicon GPU) if available, otherwise CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ITEMS_CACHE_PATH = Path(__file__).parent / "data" / "items_cache_s3.json"
CHECKPOINT_PATH = Path(__file__).parent / "data" / "checkpoint_clip.json"

# --------------- CLIP ---------------

clip_model = None
clip_processor = None


def get_clip():
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Loading CLIP ViT-L/14 on {DEVICE}...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32)
        clip_model.to(DEVICE)
        clip_model.eval()
        print("CLIP loaded.")
    return clip_processor, clip_model


def embed_images(images: list[Image.Image]) -> list[list[float]]:
    """Embed a batch of PIL images with CLIP. Returns list of 768d vectors."""
    proc, model = get_clip()
    inputs = proc(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "mps":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    # L2-normalize
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().tolist()


# --------------- Download ---------------

def download_crop(url: str) -> Image.Image | None:
    """Download a crop image from Supabase."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def download_batch(urls: list[str]) -> list[Image.Image | None]:
    """Download multiple crops in parallel."""
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        results = list(pool.map(download_crop, urls))
    return results


# --------------- Qdrant ---------------

def ensure_collection():
    r = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY}, timeout=10,
    )
    if r.status_code == 200:
        print(f"Collection '{COLLECTION}' exists")
        return
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 0},
        },
        timeout=10,
    )
    resp.raise_for_status()
    print(f"Created collection '{COLLECTION}'")


def upsert_batch(points: list[dict]):
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"points": points}, timeout=60,
    )
    resp.raise_for_status()


# --------------- Checkpoint ---------------

def load_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text())
            n = data.get("items_done", 0)
            print(f"Resuming from checkpoint: {n} items already done")
            return n
        except Exception:
            pass
    return 0


def save_checkpoint(items_done: int):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"items_done": items_done, "collection": COLLECTION}))


# --------------- Main ---------------

def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    fresh = "--fresh" in flags

    if not ITEMS_CACHE_PATH.exists():
        print(f"ERROR: {ITEMS_CACHE_PATH} not found.")
        print("Run build_index_local_s3.py first to build the items cache.")
        return

    print("Loading items cache...")
    with open(ITEMS_CACHE_PATH) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items")

    # Load CLIP model
    get_clip()

    # Ensure Qdrant collection
    ensure_collection()

    # Checkpoint
    start_from = 0 if fresh else load_checkpoint()
    total = len(items)

    if fresh:
        CHECKPOINT_PATH.unlink(missing_ok=True)

    if start_from >= total:
        print(f"All {total} items already indexed!")
        return

    if start_from > 0:
        print(f"Skipping first {start_from} items (already done)")

    remaining = items[start_from:]
    done = start_from
    t0 = time.time()
    download_fails = 0

    # Split into chunks
    chunks = [remaining[i : i + UPSERT_BATCH] for i in range(0, len(remaining), UPSERT_BATCH)]

    # Prefetch first batch
    prefetch_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    prefetch_future = None

    def start_prefetch(chunk):
        urls = [item["crop_url"] for item in chunk]
        return prefetch_pool.submit(download_batch, urls)

    if chunks:
        prefetch_future = start_prefetch(chunks[0])

    for ci, chunk in enumerate(chunks):
        # Wait for current download
        images = prefetch_future.result()

        # Start downloading NEXT batch while we embed this one
        if ci + 1 < len(chunks):
            prefetch_future = start_prefetch(chunks[ci + 1])

        # Split into items with images and items without
        valid = [(item, img) for item, img in zip(chunk, images) if img is not None]
        download_fails += len(chunk) - len(valid)

        if not valid:
            done += len(chunk)
            continue

        # Embed in sub-batches
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

        # Build Qdrant points
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

        # Upsert
        try:
            upsert_batch(points)
        except Exception as e:
            print(f"Upsert error at item {done}: {e}")
            save_checkpoint(done)
            print(f"Checkpoint saved at {done}. Re-run to resume.")
            prefetch_pool.shutdown(wait=False)
            return

        done += len(chunk)

        if done % CHECKPOINT_EVERY < UPSERT_BATCH or ci == len(chunks) - 1:
            save_checkpoint(done)

        elapsed = time.time() - t0
        rate = (done - start_from) / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] embedded+upserted | {download_fails} dl fails | {rate:.0f} items/s | ETA {eta/60:.0f}m")

    prefetch_pool.shutdown()

    print(f"\nDone! {total} items indexed in '{COLLECTION}' in {(time.time()-t0)/60:.1f}m")
    print(f"  Download failures: {download_fails}")
    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

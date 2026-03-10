"""
Build CLIP image vector index from local Fashionpedia images.

Crops garments locally (no network download), embeds with CLIP, upserts to Qdrant.

Usage:
  python -u build_clip_index_local.py /path/to/train_images
  python -u build_clip_index_local.py /path/to/train_images --fresh
"""

import io
import json
import os
import sys
import time
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
EMBED_DIM = 768
CLIP_BATCH = 128
UPSERT_BATCH = 500
CHECKPOINT_EVERY = 500

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

WEARABLE_IDS = set(range(27))

CHECKPOINT_PATH = Path(__file__).parent / "data" / "checkpoint_clip_local.json"

# --------------- CLIP ---------------

clip_model = None
clip_processor = None


def get_clip():
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Loading CLIP ViT-L/14 on {DEVICE}...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_model.to(DEVICE)
        clip_model.eval()
        print("CLIP loaded.")
    return clip_processor, clip_model


def embed_images(images: list[Image.Image]) -> list[list[float]]:
    proc, model = get_clip()
    inputs = proc(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().tolist()


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
        json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}, "optimizers_config": {"indexing_threshold": 0}},
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


# --------------- Crop ---------------

def crop_garment(img: Image.Image, bbox: list, padding: float = 0.1) -> Image.Image | None:
    x, y, w, h = bbox
    img_w, img_h = img.size
    pad_x, pad_y = w * padding, h * padding
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(img_w, int(x + w + pad_x))
    bottom = min(img_h, int(y + h + pad_y))

    if right - left < 10 or bottom - top < 10:
        return None

    crop = img.crop((left, top, right, bottom))
    max_side = max(crop.size)
    if max_side > 400:
        scale = 400 / max_side
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)

    return crop


# --------------- Main ---------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    fresh = "--fresh" in flags

    if not args:
        print("Usage: python -u build_clip_index_local.py /path/to/train_images [--fresh]")
        return

    images_dir = Path(args[0])
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} is not a directory")
        return

    jpg_count = len(list(images_dir.glob("*.jpg")))
    print(f"Found {jpg_count} JPGs in {images_dir}")

    annotations_path = Path(__file__).parent / "data" / "fashionpedia_train.json"
    if not annotations_path.exists():
        print(f"ERROR: {annotations_path} not found")
        return

    # Load items cache for metadata (crop_url, description, etc.)
    items_cache_path = Path(__file__).parent / "data" / "items_cache_s3.json"
    if not items_cache_path.exists():
        print(f"ERROR: {items_cache_path} not found")
        return

    print("Loading items cache for metadata...")
    with open(items_cache_path) as f:
        raw_items = json.load(f)
    # Map annotation ID (from crop filename) → item metadata
    items_meta = {}
    for it in raw_items:
        ann_id = int(it["crop_url"].split("/")[-1].replace(".jpg", ""))
        items_meta[ann_id] = it
    print(f"Loaded metadata for {len(items_meta)} items")

    # Load annotations
    print("Loading annotations...")
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    images_map = {img["id"]: img for img in data["images"]}

    # Build flat list matching annotations to items_cache via annotation ID
    items = []
    for ann in data["annotations"]:
        ann_id = ann["id"]
        if ann_id not in items_meta:
            continue
        if ann["category_id"] not in WEARABLE_IDS:
            continue
        if not ann.get("bbox"):
            continue
        items.append({
            "ann_id": ann_id,
            "image_id": ann["image_id"],
            "bbox": ann["bbox"],
            "meta": items_meta[ann_id],
        })

    print(f"Matched {len(items)} annotations to items cache")

    # Sort by image_id so we open each image only once
    items.sort(key=lambda x: x["image_id"])

    get_clip()
    ensure_collection()

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
    skipped = 0

    # Process in batches
    batch_images = []
    batch_metas = []
    current_img = None
    current_img_id = None

    for item in remaining:
        img_id = item["image_id"]

        # Open image if new
        if img_id != current_img_id:
            if current_img is not None:
                current_img.close()
            img_info = images_map.get(img_id, {})
            filename = img_info.get("file_name", "")
            img_path = images_dir / filename
            if img_path.exists():
                try:
                    current_img = Image.open(img_path).convert("RGB")
                except Exception:
                    current_img = None
            else:
                current_img = None
            current_img_id = img_id

        if current_img is None:
            skipped += 1
            done += 1
            continue

        crop = crop_garment(current_img, item["bbox"])
        if crop is None:
            skipped += 1
            done += 1
            continue

        batch_images.append(crop)
        batch_metas.append(item["meta"])

        # When batch is full, embed + upsert
        if len(batch_images) >= UPSERT_BATCH:
            # Embed
            all_vectors = []
            for i in range(0, len(batch_images), CLIP_BATCH):
                sub = batch_images[i : i + CLIP_BATCH]
                try:
                    vecs = embed_images(sub)
                    all_vectors.extend(vecs)
                except Exception as e:
                    print(f"  [embed] error: {e}")
                    all_vectors.extend([[0.0] * EMBED_DIM] * len(sub))

            # Build points
            points = []
            for meta, vector in zip(batch_metas, all_vectors):
                points.append({
                    "id": meta["id"],
                    "vector": vector,
                    "payload": {
                        "description": meta["description"],
                        "category": meta["category"],
                        "category_id": meta["category_id"],
                        "attributes": meta["attributes"],
                        "crop_url": meta["crop_url"],
                        "width": meta["width"],
                        "height": meta["height"],
                    },
                })

            try:
                upsert_batch(points)
            except Exception as e:
                print(f"Upsert error at item {done}: {e}")
                save_checkpoint(done)
                return

            done += len(batch_images)
            batch_images.clear()
            batch_metas.clear()

            if done % CHECKPOINT_EVERY < UPSERT_BATCH:
                save_checkpoint(done)

            elapsed = time.time() - t0
            rate = (done - start_from) / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] | {skipped} skipped | {rate:.0f} items/s | ETA {eta/60:.0f}m")

    # Final batch
    if batch_images:
        all_vectors = []
        for i in range(0, len(batch_images), CLIP_BATCH):
            sub = batch_images[i : i + CLIP_BATCH]
            try:
                vecs = embed_images(sub)
                all_vectors.extend(vecs)
            except Exception as e:
                all_vectors.extend([[0.0] * EMBED_DIM] * len(sub))

        points = []
        for meta, vector in zip(batch_metas, all_vectors):
            points.append({
                "id": meta["id"],
                "vector": vector,
                "payload": {
                    "description": meta["description"],
                    "category": meta["category"],
                    "category_id": meta["category_id"],
                    "attributes": meta["attributes"],
                    "crop_url": meta["crop_url"],
                    "width": meta["width"],
                    "height": meta["height"],
                },
            })

        try:
            upsert_batch(points)
        except Exception as e:
            print(f"Upsert error at final batch: {e}")

        done += len(batch_images)

    save_checkpoint(done)
    print(f"\nDone! {done}/{total} indexed in {(time.time()-t0)/60:.1f}m | {skipped} skipped")
    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

"""
Build Qdrant vector index from Fashionpedia train dataset.

Downloads images from Flickr, crops garments in memory, uploads crops
to Supabase Storage, embeds descriptions via Gemini, and upserts to Qdrant.

No large files saved locally — only the items_cache.json (~20MB) for resume.

Features:
  - Parallel image downloads (8 threads)
  - Parallel Gemini embedding calls (4 concurrent)
  - Checkpoint/resume: saves progress to data/checkpoint.json every 500 items
  - If it crashes, just re-run — picks up where it left off

Usage:
  1. Download annotations:
     curl -L "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" -o data/fashionpedia_train.json
  2. Set env vars in .env.local: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
  3. Run: python -u build_index.py
  4. To force full rebuild: python -u build_index.py --fresh
"""

import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

COLLECTION = "fashionpedia_v2"
EMBED_DIM = 768
EMBED_BATCH = 96        # texts per Gemini API call
EMBED_WORKERS = 4       # concurrent Gemini API calls
UPSERT_BATCH = 500      # points per Qdrant upsert
CHECKPOINT_EVERY = 500  # save progress every N items embedded+upserted
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"

SUPABASE_BUCKET = "crops"
SUPABASE_HEADERS = {
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "apikey": SUPABASE_SERVICE_KEY,
}

WEARABLE_IDS = set(range(27))

FASHIONPEDIA_ATTRIBUTES = {
    0: "floral", 1: "graphic", 2: "striped", 3: "plain", 4: "lattice",
    5: "spotted", 6: "checked", 7: "solid color",
    8: "denim", 9: "chiffon", 10: "corduroy", 11: "cotton",
    12: "faux fur", 13: "knit", 14: "lace", 15: "leather",
    16: "linen", 17: "mesh", 18: "nylon", 19: "satin",
    20: "sequined", 21: "silk", 22: "suede", 23: "velvet", 24: "wool",
    25: "long sleeve", 26: "short sleeve", 27: "sleeveless",
    28: "maxi length", 29: "midi length", 30: "mini length",
    31: "crew neckline", 32: "v-neckline", 33: "turtleneck",
    34: "sweetheart neckline", 35: "straight fit", 36: "loose fit", 37: "tight fit",
}

CHECKPOINT_PATH = Path(__file__).parent / "data" / "checkpoint.json"
ITEMS_CACHE_PATH = Path(__file__).parent / "data" / "items_cache.json"


# --------------- Supabase Storage ---------------

def ensure_bucket():
    """Create the crops bucket if it doesn't exist."""
    r = requests.get(
        f"{SUPABASE_URL}/storage/v1/bucket/{SUPABASE_BUCKET}",
        headers=SUPABASE_HEADERS,
        timeout=10,
    )
    if r.status_code == 200:
        print(f"Supabase bucket '{SUPABASE_BUCKET}' exists")
        return

    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/bucket",
        headers={**SUPABASE_HEADERS, "Content-Type": "application/json"},
        json={"id": SUPABASE_BUCKET, "name": SUPABASE_BUCKET, "public": True},
        timeout=10,
    )
    resp.raise_for_status()
    print(f"Created Supabase bucket '{SUPABASE_BUCKET}'")


def upload_crop_to_supabase(filename: str, jpeg_bytes: bytes) -> str:
    """Upload JPEG bytes to Supabase Storage. Returns public URL."""
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}",
        headers={
            **SUPABASE_HEADERS,
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        },
        data=jpeg_bytes,
        timeout=30,
    )
    resp.raise_for_status()
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"


def check_crop_exists(filename: str) -> str | None:
    """Check if crop already exists in Supabase. Returns public URL if so."""
    url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
    resp = requests.head(url, timeout=10)
    if resp.status_code == 200:
        return url
    return None


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


# --------------- Gemini Embedding ---------------

def _embed_one_batch(texts: list[str]) -> list[list[float]]:
    body = {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": EMBED_DIM,
            }
            for t in texts
        ]
    }
    try:
        resp = requests.post(GEMINI_EMBED_URL, json=body, timeout=60)
        resp.raise_for_status()
        return [e["values"] for e in resp.json()["embeddings"]]
    except Exception:
        time.sleep(30)
        resp = requests.post(GEMINI_EMBED_URL, json=body, timeout=60)
        resp.raise_for_status()
        return [e["values"] for e in resp.json()["embeddings"]]


def embed_texts_parallel(texts: list[str]) -> list[list[float]]:
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    results: list[list[list[float]]] = [[] for _ in batches]

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_embed_one_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [embed] batch {idx} failed permanently: {e}")
                results[idx] = [[0.0] * EMBED_DIM] * len(batches[idx])

    return [vec for batch_vecs in results for vec in batch_vecs]


# --------------- Qdrant ---------------

def ensure_collection():
    r = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY},
        timeout=10,
    )
    if r.status_code == 200:
        print(f"Collection '{COLLECTION}' exists, will upsert into it")
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
        json={"points": points},
        timeout=60,
    )
    resp.raise_for_status()


# --------------- Image Download + Crop (all in memory) ---------------

def download_image(original_url: str) -> Image.Image | None:
    if not original_url:
        return None
    try:
        resp = requests.get(original_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except Exception:
        return None


def crop_and_upload(img: Image.Image, bbox: list, ann_id: int, padding: float = 0.1) -> tuple[str, int, int] | None:
    """Crop garment from image, upload to Supabase, return (url, w, h)."""
    crop_filename = f"{ann_id}.jpg"

    # Check if already uploaded
    existing_url = check_crop_exists(crop_filename)
    if existing_url:
        # We don't know dimensions without downloading — crop anyway for metadata
        pass  # Fall through to crop for dimensions, but skip upload

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

    crop_w, crop_h = crop.size

    if existing_url:
        return existing_url, crop_w, crop_h

    # Encode to JPEG bytes in memory
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    try:
        url = upload_crop_to_supabase(crop_filename, jpeg_bytes)
        return url, crop_w, crop_h
    except Exception as e:
        print(f"  [upload] {crop_filename} failed: {e}")
        return None


# --------------- Build Items ---------------

def build_items(annotations_path: str) -> list[dict]:
    """Parse annotations, download images, crop garments, upload to Supabase."""
    if ITEMS_CACHE_PATH.exists():
        print(f"Loading cached items from {ITEMS_CACHE_PATH}...")
        with open(ITEMS_CACHE_PATH) as f:
            items = json.load(f)
        print(f"Loaded {len(items)} items from cache")
        return items

    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}
    images_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    img_anns: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] in WEARABLE_IDS and ann.get("bbox"):
            img_anns.setdefault(ann["image_id"], []).append(ann)

    total_images = len(img_anns)
    print(f"Processing {total_images} images...")

    items = []
    skipped = 0
    processed = 0
    image_ids = list(img_anns.keys())
    dl_batch = 20
    t0 = time.time()

    for batch_start in range(0, len(image_ids), dl_batch):
        batch_ids = image_ids[batch_start : batch_start + dl_batch]

        # Download concurrently from Flickr original URLs
        downloaded: dict[int, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for img_id in batch_ids:
                img_info = images_map.get(img_id, {})
                url = img_info.get("original_url", "")
                if url:
                    futures[pool.submit(download_image, url)] = img_id
            for future in as_completed(futures):
                img_id = futures[future]
                result = future.result()
                if result:
                    downloaded[img_id] = result

        # Crop and upload
        for img_id in batch_ids:
            img = downloaded.get(img_id)
            if not img:
                skipped += len(img_anns[img_id])
                continue

            for ann in img_anns[img_id]:
                crop_result = crop_and_upload(img, ann["bbox"], ann["id"])
                if not crop_result:
                    skipped += 1
                    continue
                crop_url, crop_w, crop_h = crop_result
                cat_id = ann["category_id"]
                cat_name = categories.get(cat_id, "clothing")
                attr_ids = ann.get("attribute_ids", [])
                attr_names = [attributes.get(a, FASHIONPEDIA_ATTRIBUTES.get(a, "")) for a in attr_ids]
                attr_names = [a for a in attr_names if a]
                desc_parts = []
                if attr_names:
                    desc_parts.append(", ".join(attr_names))
                desc_parts.append(cat_name)
                items.append({
                    "id": len(items),
                    "description": " ".join(desc_parts),
                    "category": cat_name,
                    "category_id": cat_id,
                    "attributes": attr_names,
                    "crop_url": crop_url,
                    "width": crop_w,
                    "height": crop_h,
                })

            img.close()
            processed += 1

        if processed % 200 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_images - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{total_images}] {len(items)} items | {rate:.0f} img/s | ETA {eta/60:.0f}m")

    print(f"Built {len(items)} items ({skipped} skipped) in {(time.time()-t0)/60:.1f}m")

    print(f"Saving items cache to {ITEMS_CACHE_PATH}...")
    ITEMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ITEMS_CACHE_PATH, "w") as f:
        json.dump(items, f)

    return items


# --------------- Main ---------------

def main():
    fresh = "--fresh" in sys.argv

    annotations_path = Path(__file__).parent / "data" / "fashionpedia_train.json"
    if not annotations_path.exists():
        print("ERROR: Download annotations first:")
        print('  curl -L "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" -o data/fashionpedia_train.json')
        return

    if fresh:
        print("--fresh: clearing checkpoint and items cache")
        CHECKPOINT_PATH.unlink(missing_ok=True)
        ITEMS_CACHE_PATH.unlink(missing_ok=True)

    # Phase 0: Ensure Supabase bucket exists
    ensure_bucket()

    # Phase 1: Build items (download + crop + upload to Supabase)
    items = build_items(str(annotations_path))
    if not items:
        print("No items to index")
        return

    # Phase 2: Ensure Qdrant collection exists
    ensure_collection()

    # Phase 3: Embed + upsert with checkpoint resume
    start_from = 0 if fresh else load_checkpoint()
    total = len(items)

    if start_from >= total:
        print(f"All {total} items already indexed!")
        return

    if start_from > 0:
        print(f"Skipping first {start_from} items (already done)")

    remaining = items[start_from:]
    done = start_from
    t0 = time.time()

    for chunk_start in range(0, len(remaining), UPSERT_BATCH):
        chunk = remaining[chunk_start : chunk_start + UPSERT_BATCH]
        texts = [item["description"] for item in chunk]

        try:
            vectors = embed_texts_parallel(texts)
        except Exception as e:
            print(f"FATAL embed error at item {done}: {e}")
            save_checkpoint(done)
            print(f"Checkpoint saved at {done}. Re-run to resume.")
            return

        points = []
        for item, vector in zip(chunk, vectors):
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
            save_checkpoint(done)
            print(f"Checkpoint saved at {done}. Re-run to resume.")
            return

        done += len(chunk)

        if done % CHECKPOINT_EVERY < UPSERT_BATCH or chunk_start + UPSERT_BATCH >= len(remaining):
            save_checkpoint(done)

        elapsed = time.time() - t0
        rate = (done - start_from) / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] embedded+upserted | {rate:.0f} items/s | ETA {eta/60:.0f}m")

    print(f"\nDone! {total} items indexed in '{COLLECTION}' in {(time.time()-t0)/60:.1f}m")
    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

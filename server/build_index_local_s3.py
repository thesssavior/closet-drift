"""
Build Qdrant vector index from Fashionpedia using local S3 images.

Reads full-res images from extracted train2020.zip, crops garments,
uploads to Supabase Storage, embeds via Gemini, upserts to Qdrant.

Usage:
  1. Download & extract: https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
  2. Run: python -u build_index_local_s3.py /path/to/train2020
  3. Resume after crash: just re-run the same command
  4. Fresh start: python -u build_index_local_s3.py /path/to/train2020 --fresh
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
EMBED_BATCH = 96
UPSERT_BATCH = 500
CHECKPOINT_EVERY = 500
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

CHECKPOINT_PATH = Path(__file__).parent / "data" / "checkpoint_s3.json"
ITEMS_CACHE_PATH = Path(__file__).parent / "data" / "items_cache_s3.json"


# --------------- Supabase Storage ---------------

def ensure_bucket():
    r = requests.get(f"{SUPABASE_URL}/storage/v1/bucket/{SUPABASE_BUCKET}", headers=SUPABASE_HEADERS, timeout=10)
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


def upload_crop(filename: str, jpeg_bytes: bytes) -> str:
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}",
        headers={**SUPABASE_HEADERS, "Content-Type": "image/jpeg", "x-upsert": "true"},
        data=jpeg_bytes, timeout=30,
    )
    resp.raise_for_status()
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"


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


# --------------- Gemini Embedding (sequential with retry) ---------------

def _embed_one_batch(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    body = {
        "requests": [
            {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}, "outputDimensionality": EMBED_DIM}
            for t in texts
        ]
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_EMBED_URL, json=body, timeout=60)
            if resp.status_code == 429:
                wait = min(30 * (2 ** attempt), 120)
                print(f"    rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return [e["values"] for e in resp.json()["embeddings"]]
        except requests.exceptions.HTTPError:
            raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            raise
    raise Exception(f"Failed after {max_retries} retries (rate limited)")


def embed_texts_sequential(texts: list[str]) -> list[list[float]]:
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    all_vecs = []
    for i, batch in enumerate(batches):
        try:
            vecs = _embed_one_batch(batch)
            all_vecs.extend(vecs)
        except Exception as e:
            print(f"  [embed] batch {i} failed permanently: {e}")
            all_vecs.extend([[0.0] * EMBED_DIM] * len(batch))
        if i < len(batches) - 1:
            time.sleep(1)
    return all_vecs


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


# --------------- Crop + Upload ---------------

UPLOAD_WORKERS = 16

def crop_garment(img: Image.Image, bbox: list, ann_id: int, padding: float = 0.1) -> tuple[str, bytes, int, int] | None:
    """Crop garment from image. Returns (filename, jpeg_bytes, w, h) or None."""
    crop_filename = f"{ann_id}.jpg"

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

    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    return crop_filename, buf.getvalue(), crop_w, crop_h


def _upload_one(args: tuple) -> tuple[str, int, int] | None:
    """Upload a single crop. Used by thread pool."""
    filename, jpeg_bytes, crop_w, crop_h = args
    try:
        url = upload_crop(filename, jpeg_bytes)
        return url, crop_w, crop_h
    except Exception as e:
        print(f"  [upload] {filename} failed: {e}")
        return None


# --------------- Build Items ---------------

PHASE1_CHECKPOINT_PATH = Path(__file__).parent / "data" / "phase1_checkpoint_s3.json"


def _save_phase1_checkpoint(done_image_ids: set, items: list[dict]):
    PHASE1_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE1_CHECKPOINT_PATH.write_text(json.dumps({
        "done_image_ids": list(done_image_ids),
        "items": items,
    }))


def build_items(annotations_path: str, images_dir: Path) -> list[dict]:
    if ITEMS_CACHE_PATH.exists():
        print(f"Loading cached items from {ITEMS_CACHE_PATH}...")
        with open(ITEMS_CACHE_PATH) as f:
            items = json.load(f)
        print(f"Loaded {len(items)} items from cache")
        return items

    # Load Phase 1 checkpoint if exists
    done_image_ids: set[int] = set()
    items: list[dict] = []
    if PHASE1_CHECKPOINT_PATH.exists():
        try:
            ckpt = json.loads(PHASE1_CHECKPOINT_PATH.read_text())
            done_image_ids = set(ckpt["done_image_ids"])
            items = ckpt["items"]
            print(f"Resuming Phase 1: {len(done_image_ids)} images done, {len(items)} items so far")
        except Exception:
            pass

    print(f"Loading annotations...")
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}
    images_map = {img["id"]: img for img in data["images"]}

    img_anns: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann["category_id"] in WEARABLE_IDS and ann.get("bbox"):
            img_anns.setdefault(ann["image_id"], []).append(ann)

    # Filter out already-done images
    image_ids = [iid for iid in img_anns.keys() if iid not in done_image_ids]
    total_images = len(img_anns)
    already_done = total_images - len(image_ids)
    print(f"Processing {len(image_ids)} remaining images ({already_done} already done) from {images_dir}...")

    skipped = 0
    processed = already_done
    uploaded = len(items)
    missing = 0
    t0 = time.time()
    batch_size = 50

    for batch_start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[batch_start : batch_start + batch_size]

        # Phase A: Crop all garments from this batch (fast, local)
        pending_uploads = []
        batch_done_ids = []
        for img_id in batch_ids:
            anns = img_anns[img_id]
            img_info = images_map.get(img_id, {})
            filename = img_info.get("file_name", "")
            img_path = images_dir / filename

            if not img_path.exists():
                skipped += len(anns)
                missing += 1
                batch_done_ids.append(img_id)
                continue

            try:
                img = Image.open(img_path)
            except Exception:
                skipped += len(anns)
                batch_done_ids.append(img_id)
                continue

            for ann in anns:
                crop_result = crop_garment(img, ann["bbox"], ann["id"])
                if not crop_result:
                    skipped += 1
                    continue
                crop_filename, jpeg_bytes, crop_w, crop_h = crop_result
                cat_id = ann["category_id"]
                cat_name = categories.get(cat_id, "clothing")
                attr_ids = ann.get("attribute_ids", [])
                attr_names = [attributes.get(a, FASHIONPEDIA_ATTRIBUTES.get(a, "")) for a in attr_ids]
                attr_names = [a for a in attr_names if a]
                pending_uploads.append((crop_filename, jpeg_bytes, crop_w, crop_h, cat_id, cat_name, attr_names))

            img.close()
            processed += 1
            batch_done_ids.append(img_id)

        # Phase B: Upload all crops in parallel
        if pending_uploads:
            upload_args = [(fn, jb, cw, ch) for fn, jb, cw, ch, _, _, _ in pending_uploads]
            with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
                results = list(pool.map(_upload_one, upload_args))

            for (crop_filename, jpeg_bytes, crop_w, crop_h, cat_id, cat_name, attr_names), result in zip(pending_uploads, results):
                if result is None:
                    skipped += 1
                    continue
                crop_url, crop_w, crop_h = result
                uploaded += 1
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

        # Track done images
        done_image_ids.update(batch_done_ids)

        # Checkpoint every 500 images
        if processed % 500 < batch_size:
            _save_phase1_checkpoint(done_image_ids, items)

        elapsed = time.time() - t0
        imgs_this_run = processed - already_done
        rate = imgs_this_run / elapsed if elapsed > 0 else 0
        remaining = len(image_ids) - (batch_start + len(batch_ids))
        eta = remaining / rate if rate > 0 else 0
        print(f"  [{processed}/{total_images}] {uploaded} crops | {skipped} skipped | {missing} missing | {rate:.1f} img/s | ETA {eta/60:.0f}m")

    print(f"Built {len(items)} items ({skipped} skipped, {missing} missing) in {(time.time()-t0)/60:.1f}m")

    ITEMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving items cache...")
    with open(ITEMS_CACHE_PATH, "w") as f:
        json.dump(items, f)

    # Clean up phase 1 checkpoint
    PHASE1_CHECKPOINT_PATH.unlink(missing_ok=True)

    return items


# --------------- Main ---------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    fresh = "--fresh" in flags

    if not args:
        print("Usage: python -u build_index_local_s3.py /path/to/train2020 [--fresh]")
        print("  Download & extract: https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip")
        return

    images_dir = Path(args[0])
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} is not a directory")
        return

    # Quick sanity check
    jpg_count = len(list(images_dir.glob("*.jpg")))
    print(f"Found {jpg_count} JPGs in {images_dir}")
    if jpg_count == 0:
        print("ERROR: No JPGs found. Make sure you point to the extracted train2020 folder.")
        return

    annotations_path = Path(__file__).parent / "data" / "fashionpedia_train.json"
    if not annotations_path.exists():
        print("ERROR: Download annotations first:")
        print('  curl -L "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" -o data/fashionpedia_train.json')
        return

    if fresh:
        print("--fresh: clearing checkpoint and items cache")
        CHECKPOINT_PATH.unlink(missing_ok=True)
        ITEMS_CACHE_PATH.unlink(missing_ok=True)
        PHASE1_CHECKPOINT_PATH.unlink(missing_ok=True)

    # Phase 0
    ensure_bucket()

    # Phase 1: Build items from local images
    items = build_items(str(annotations_path), images_dir)
    if not items:
        print("No items to index")
        return

    # Phase 2
    ensure_collection()

    # Phase 3: Embed + upsert
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
            vectors = embed_texts_sequential(texts)
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

"""
Build Qdrant vector index from DeepFashion2 dataset.

Crops garments from images using bbox annotations, generates text descriptions,
embeds with Gemini (768-dim), and upserts to Qdrant.

Features:
  - Parallel image cropping (8 threads)
  - Parallel Gemini embedding (4 concurrent API calls)
  - Checkpoint/resume: saves progress, re-run to continue after crash
  - Items cache: skip parsing on re-run

Expected directory structure:
  server/data/deepfashion2/
    train/
      image/
        000001.jpg
        000002.jpg
        ...
      annos/
        000001.json
        000002.json
        ...
    validation/
      image/
      annos/

Usage:
  1. Download DeepFashion2 and extract to server/data/deepfashion2/
  2. Set env vars in .env.local: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
  3. Run: python3 build_index_deepfashion.py
  4. To force rebuild: python3 build_index_deepfashion.py --fresh
"""

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

COLLECTION = "fashionpedia_v2"
EMBED_DIM = 768
EMBED_BATCH = 96
EMBED_WORKERS = 4
UPSERT_BATCH = 500
CHECKPOINT_EVERY = 500
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"

DATA_DIR = Path(__file__).parent / "data" / "deepfashion2"
CROPS_DIR = Path(__file__).parent / "static" / "crops"
CHECKPOINT_PATH = Path(__file__).parent / "data" / "df2_checkpoint.json"
ITEMS_CACHE_PATH = Path(__file__).parent / "data" / "df2_items_cache.json"

# DeepFashion2 categories (1-indexed in annotations)
DF2_CATEGORIES = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}

# Map DeepFashion2 categories → Fashionpedia category_ids (for filtering compatibility)
DF2_TO_FASHIONPEDIA = {
    1: 1,   # short sleeve top → top
    2: 1,   # long sleeve top → top
    3: 4,   # short sleeve outwear → jacket
    4: 4,   # long sleeve outwear → jacket
    5: 5,   # vest → vest
    6: 0,   # sling → shirt/blouse
    7: 7,   # shorts → shorts
    8: 6,   # trousers → pants
    9: 8,   # skirt → skirt
    10: 10,  # short sleeve dress → dress
    11: 10,  # long sleeve dress → dress
    12: 10,  # vest dress → dress
    13: 10,  # sling dress → dress
}


# --------------- Checkpoint ---------------

def load_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text()).get("items_done", 0)
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
                print(f"  [embed] batch {idx} failed: {e}")
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


# --------------- Cropping ---------------

def crop_garment(image_path: Path, bbox: list, item_id: str, padding: float = 0.1) -> tuple[str, int, int] | None:
    """Crop garment. bbox is [x1, y1, x2, y2]. Returns (filename, w, h)."""
    crop_filename = f"df2_{item_id}.jpg"
    crop_path = CROPS_DIR / crop_filename
    if crop_path.exists():
        try:
            with Image.open(crop_path) as existing:
                return crop_filename, existing.size[0], existing.size[1]
        except Exception:
            pass

    try:
        img = Image.open(image_path)
    except Exception:
        return None

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    img_w, img_h = img.size
    pad_x, pad_y = w * padding, h * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(img_w, int(x2 + pad_x))
    bottom = min(img_h, int(y2 + pad_y))

    if right - left < 10 or bottom - top < 10:
        return None

    crop = img.crop((left, top, right, bottom))
    max_side = max(crop.size)
    if max_side > 400:
        scale = 400 / max_side
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)

    crop.save(crop_path, "JPEG", quality=85)
    return crop_filename, crop.size[0], crop.size[1]


# --------------- Build Items ---------------

def process_annotation_file(args: tuple) -> list[dict]:
    """Process a single annotation JSON file. Returns list of items."""
    anno_path, image_dir, start_id = args
    items = []

    try:
        with open(anno_path) as f:
            data = json.load(f)
    except Exception:
        return items

    # Image filename matches annotation filename but with .jpg
    img_name = anno_path.stem + ".jpg"
    img_path = image_dir / img_name
    if not img_path.exists():
        return items

    # Each key like "item1", "item2", ... is a clothing item
    for key, item in data.items():
        if not key.startswith("item"):
            continue

        cat_id = item.get("category_id")
        if not cat_id or cat_id not in DF2_CATEGORIES:
            continue

        bbox = item.get("bounding_box")
        if not bbox or len(bbox) < 4:
            continue

        # Skip heavily occluded items
        occlusion = item.get("occlusion", 1)
        if occlusion > 2:
            continue

        item_id = f"{anno_path.stem}_{key}"
        crop_result = crop_garment(img_path, bbox, item_id)
        if not crop_result:
            continue

        crop_filename, crop_w, crop_h = crop_result
        cat_name = DF2_CATEGORIES[cat_id]
        fashionpedia_cat = DF2_TO_FASHIONPEDIA.get(cat_id, 0)

        # Build description from category + style info
        style = item.get("style", 0)
        style_str = f"style {style}" if style and style > 0 else ""
        description = f"{style_str} {cat_name}".strip()

        items.append({
            "description": description,
            "category": cat_name,
            "category_id": fashionpedia_cat,
            "attributes": [cat_name],
            "crop_filename": crop_filename,
            "width": crop_w,
            "height": crop_h,
        })

    return items


def build_items() -> list[dict]:
    """Parse all DeepFashion2 annotations, crop garments. Caches to disk."""
    if ITEMS_CACHE_PATH.exists():
        print(f"Loading cached items from {ITEMS_CACHE_PATH}...")
        with open(ITEMS_CACHE_PATH) as f:
            items = json.load(f)
        print(f"Loaded {len(items)} items from cache")
        return items

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []

    # Process both train and validation splits
    for split in ["train", "validation"]:
        anno_dir = DATA_DIR / split / "annos"
        image_dir = DATA_DIR / split / "image"

        if not anno_dir.exists():
            print(f"Skipping {split}: {anno_dir} not found")
            continue

        anno_files = sorted(anno_dir.glob("*.json"))
        total = len(anno_files)
        print(f"Processing {split}: {total} annotation files...")

        t0 = time.time()
        processed = 0

        # Process in parallel batches
        batch_size = 100
        for batch_start in range(0, total, batch_size):
            batch_files = anno_files[batch_start : batch_start + batch_size]
            args_list = [(f, image_dir, 0) for f in batch_files]

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(process_annotation_file, args): args for args in args_list}
                for future in as_completed(futures):
                    result = future.result()
                    for item in result:
                        item["id"] = len(all_items)
                        all_items.append(item)

            processed += len(batch_files)
            if processed % 1000 == 0 or batch_start + batch_size >= total:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"  [{split}] [{processed}/{total}] {len(all_items)} items | {rate:.0f} files/s | ETA {eta/60:.0f}m")

    print(f"\nBuilt {len(all_items)} total items")

    # Cache to disk
    print(f"Saving items cache to {ITEMS_CACHE_PATH}...")
    with open(ITEMS_CACHE_PATH, "w") as f:
        json.dump(all_items, f)

    return all_items


# --------------- Main ---------------

def main():
    fresh = "--fresh" in sys.argv

    if not DATA_DIR.exists():
        print(f"ERROR: DeepFashion2 data not found at {DATA_DIR}")
        print("Expected structure:")
        print("  server/data/deepfashion2/train/image/  + annos/")
        print("  server/data/deepfashion2/validation/image/  + annos/")
        return

    if fresh:
        print("--fresh: clearing checkpoint and items cache")
        CHECKPOINT_PATH.unlink(missing_ok=True)
        ITEMS_CACHE_PATH.unlink(missing_ok=True)

    # Phase 1: Build items (parse + crop)
    items = build_items()
    if not items:
        print("No items to index")
        return

    # Phase 2: Ensure collection
    ensure_collection()

    # Create payload index for category filtering
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qc.create_payload_index(
            collection_name=COLLECTION,
            field_name="category_id",
            field_schema="integer",
        )
        print("Created payload index on category_id")
    except Exception:
        pass

    # Phase 3: Embed + upsert with checkpoint resume
    start_from = 0 if fresh else load_checkpoint()
    total = len(items)

    if start_from >= total:
        print(f"All {total} items already indexed!")
        return

    if start_from > 0:
        print(f"Resuming from checkpoint: skipping first {start_from} items")

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
                    "crop_filename": item["crop_filename"],
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

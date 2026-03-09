import io
import json
import os
import base64
import hashlib
from pathlib import Path

from dotenv import load_dotenv

# Load .env.local from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

import numpy as np
import requests as http_requests
import torch
from torchvision.ops import nms
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Sam2Processor,
    Sam2Model,
)

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve cropped garment images
_static_dir = Path(__file__).parent / "static" / "crops"
if _static_dir.exists():
    app.mount("/crops", StaticFiles(directory=str(_static_dir)), name="crops")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"

qdrant: QdrantClient | None = None
QDRANT_COLLECTION = "fashionpedia_v2"

# YOLOS-Fashionpedia category labels (46 classes)
YOLOS_LABELS = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater", 3: "cardigan",
    4: "jacket", 5: "vest", 6: "pants", 7: "shorts", 8: "skirt", 9: "coat",
    10: "dress", 11: "jumpsuit", 12: "cape", 13: "glasses", 14: "hat",
    15: "headband, head covering", 16: "tie", 17: "glove", 18: "watch", 19: "belt",
    20: "leg warmer", 21: "tights, stockings", 22: "sock", 23: "shoe",
    24: "bag, wallet", 25: "scarf", 26: "umbrella",
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette", 31: "sleeve",
    32: "pocket", 33: "neckline", 34: "buckle", 35: "zipper", 36: "applique",
    37: "bead", 38: "bow", 39: "flower", 40: "fringe", 41: "ribbon",
    42: "rivet", 43: "ruffle", 44: "sequin", 45: "tassel",
}

# Only keep main, large garment categories
YOLOS_CLOTHING_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 23, 24}
#  0 shirt/blouse, 1 top/t-shirt, 2 sweater, 3 cardigan, 4 jacket, 5 vest,
#  6 pants, 7 shorts, 8 skirt, 9 coat, 10 dress, 11 jumpsuit,
# 21 tights/stockings, 23 shoe, 24 bag/wallet

# YOLOS category → search-friendly name
YOLOS_SEARCH_NAMES = {
    0: "shirt blouse", 1: "top t-shirt sweatshirt", 2: "sweater", 3: "cardigan",
    4: "jacket", 5: "vest", 6: "pants", 7: "shorts", 8: "skirt", 9: "coat",
    10: "dress", 11: "jumpsuit", 12: "cape", 13: "glasses sunglasses", 14: "hat",
    15: "headband head covering", 16: "tie", 17: "glove", 18: "watch", 19: "belt",
    20: "leg warmer", 21: "tights stockings", 22: "sock", 23: "shoe shoes",
    24: "bag wallet", 25: "scarf", 26: "umbrella",
}

# YOLOS category → Fashionpedia category IDs for Qdrant filtering
YOLOS_TO_FASHIONPEDIA = {
    0: [0],       # shirt/blouse
    1: [1],       # top/t-shirt/sweatshirt
    2: [2],       # sweater
    3: [3],       # cardigan
    4: [4],       # jacket
    5: [5],       # vest
    6: [6],       # pants
    7: [7],       # shorts
    8: [8],       # skirt
    9: [9],       # coat
    10: [10],     # dress
    11: [11],     # jumpsuit
    12: [12],     # cape
    13: [13],     # glasses
    14: [14],     # hat
    15: [15],     # headband
    16: [16],     # tie
    17: [17],     # glove
    18: [18],     # watch
    19: [19],     # belt
    20: [20],     # leg warmer
    21: [21],     # tights
    22: [22],     # sock
    23: [23],     # shoe
    24: [24],     # bag
    25: [25],     # scarf
    26: [26],     # umbrella
}

# YOLOS + SAM2
yolos_processor = None
yolos_model = None
sam2_processor = None
sam2_model = None

# Cache: image_hash -> { instances, orig_w, orig_h, image }
# instances = list of { category_id, category_name, mask (np.ndarray), bbox }
seg_cache: dict[str, dict] = {}

YOLOS_CONFIDENCE = 0.25  # detection threshold


def get_yolos():
    global yolos_processor, yolos_model
    if yolos_processor is None:
        print("Loading YOLOS-Fashionpedia...")
        yolos_processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        yolos_model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
        yolos_model.eval()
        print("YOLOS loaded.")
    return yolos_processor, yolos_model


def get_sam2():
    global sam2_processor, sam2_model
    if sam2_processor is None:
        print("Loading SAM2 tiny...")
        sam2_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
        sam2_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        sam2_model.eval()
        print("SAM2 loaded.")
    return sam2_processor, sam2_model


def detect_and_segment(img: Image.Image) -> list[dict]:
    """Run YOLOS detection + SAM2 segmentation. Returns list of instances."""
    yproc, ymdl = get_yolos()
    sproc, smdl = get_sam2()

    # Step 1: YOLOS detection
    inputs = yproc(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = ymdl(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])  # (h, w)
    results = yproc.post_process_object_detection(outputs, threshold=YOLOS_CONFIDENCE, target_sizes=target_sizes)[0]

    boxes = results["boxes"]  # (N, 4) in xyxy format
    scores = results["scores"]
    labels = results["labels"]

    # Filter to main apparel only (skip parts like collar, sleeve, pocket)
    keep = [i for i in range(len(labels)) if labels[i].item() in YOLOS_CLOTHING_IDS]
    if not keep:
        return []

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # NMS: remove duplicate overlapping detections (IoU > 0.5 → keep higher score)
    nms_keep = nms(boxes, scores, iou_threshold=0.5)
    boxes = boxes[nms_keep]
    scores = scores[nms_keep]
    labels = labels[nms_keep]

    detected = [(YOLOS_LABELS.get(l.item(), "?"), f"{s:.2f}") for l, s in zip(labels, scores)]
    print(f"  [yolos] detected ({len(labels)} after NMS): {detected}")

    # Step 2: SAM2 segmentation — process each box individually to avoid batch shape issues
    instances = []
    w, h = img.size
    for i in range(len(labels)):
        cat_id = labels[i].item()
        box = boxes[i].tolist()
        single_box = [[box]]  # [[[x1, y1, x2, y2]]]
        sam_inputs = sproc(images=img, input_boxes=single_box, return_tensors="pt")
        with torch.no_grad():
            sam_outputs = smdl(**sam_inputs)
        mask_pred = sam_outputs.pred_masks.cpu()  # (1, 1, num_masks, H, W)
        iou_scores = sam_outputs.iou_scores.cpu()  # (1, 1, num_masks)
        best_idx = iou_scores[0, 0].argmax().item()
        mask_tensor = mask_pred[0, 0, best_idx]  # (H, W) logits
        # Resize to original image size
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask = (mask_resized > 0).numpy().astype(np.uint8)

        # Clip mask to bounding box — prevents SAM2 from bleeding into face/skin
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bbox_mask = np.zeros_like(mask)
        bbox_mask[y1:y2, x1:x2] = 1
        mask = mask * bbox_mask

        mask_pixels = int(mask.sum())
        print(f"    [{i}] {YOLOS_LABELS.get(cat_id, '?')} score={scores[i]:.2f} bbox=({x1},{y1},{x2},{y2}) mask_pixels={mask_pixels}")

        instances.append({
            "category_id": cat_id,
            "category_name": YOLOS_LABELS.get(cat_id, "unknown"),
            "mask": mask,
            "bbox": {"x": int(box[0]), "y": int(box[1]), "w": int(box[2] - box[0]), "h": int(box[3] - box[1])},
            "score": scores[i].item(),
        })

    return instances


def build_seg_map_from_instances(instances: list[dict], w: int, h: int) -> np.ndarray:
    """Build an instance-based seg_map. Each detected item gets a unique ID (1, 2, 3...).
    0 = background. Sorted by score ascending so higher-confidence paints last."""
    seg_map = np.zeros((h, w), dtype=np.uint8)
    sorted_inst = sorted(instances, key=lambda x: x["score"])
    print(f"  [seg_map] building {w}x{h}, {len(sorted_inst)} instances:")
    for idx, inst in enumerate(sorted_inst):
        instance_id = idx + 1  # 1-based, 0 = background
        inst["instance_id"] = instance_id
        before = int((seg_map > 0).sum())
        seg_map[inst["mask"] > 0] = instance_id
        after = int((seg_map > 0).sum())
        added = after - before
        overwritten = int(inst["mask"].sum()) - added
        print(f"    id={instance_id} {inst['category_name']} score={inst['score']:.2f} pixels={int(inst['mask'].sum())} added={added} overwritten={overwritten}")
    unique_ids = set(np.unique(seg_map)) - {0}
    print(f"  [seg_map] final unique IDs: {unique_ids}")
    return seg_map


def image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def dominant_color_name(img: Image.Image, mask: np.ndarray) -> str:
    """Get the dominant color name from masked region of image."""
    arr = np.array(img)
    pixels = arr[mask > 0]
    if len(pixels) == 0:
        return ""

    # Subsample for speed
    if len(pixels) > 1000:
        idx = np.random.choice(len(pixels), 1000, replace=False)
        pixels = pixels[idx]

    avg = pixels.mean(axis=0).astype(int)
    r, g, b = int(avg[0]), int(avg[1]), int(avg[2])

    # Map to basic color names
    hsv_img = Image.new("RGB", (1, 1), (r, g, b)).convert("HSV")
    h, s, v = hsv_img.getpixel((0, 0))

    if v < 40:
        return "black"
    if s < 25 and v > 200:
        return "white"
    if s < 30:
        return "gray"

    # Hue-based (0-255 scale in PIL)
    hue = h * 2  # convert to 0-360
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 160:
        return "green"
    if hue < 250:
        return "blue"
    if hue < 300:
        return "purple"
    return "pink"


def embed_query(text: str, output_dimensionality: int | None = None) -> list[float]:
    """Embed a single query using Gemini."""
    body: dict = {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}}
    if output_dimensionality:
        body["outputDimensionality"] = output_dimensionality
    resp = http_requests.post(GEMINI_EMBED_URL, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def _point_to_product(point) -> dict:
    p = point.payload
    image_url = p.get("crop_url", "")
    if not image_url:
        crop = p.get("crop_filename", "")
        image_url = f"/crops/{crop}" if crop else ""
    return {
        "id": point.id,
        "name": p.get("description", ""),
        "price": "",
        "image": image_url,
        "brand": ", ".join(p.get("attributes", [])[:3]),
        "link": "",
        "width": p.get("width", 0),
        "height": p.get("height", 0),
    }


def search_qdrant(
    query: str,
    limit: int = 100,
    offset: int = 0,
    category_ids: list[int] | None = None,
) -> list[dict]:
    """Vector search on Qdrant using Gemini embeddings."""
    if not qdrant or not GEMINI_API_KEY:
        return []

    try:
        vector = embed_query(query, output_dimensionality=768)
        query_filter = None
        if category_ids:
            query_filter = Filter(
                must=[FieldCondition(key="category_id", match=MatchAny(any=category_ids))]
            )
        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            offset=offset,
            query_filter=query_filter,
        )
        return [_point_to_product(point) for point in results.points]
    except Exception as e:
        print(f"[search] error: {e}")
        return []


@app.on_event("startup")
async def startup():
    global qdrant
    get_yolos()
    get_sam2()

    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    if qdrant_url and qdrant_key:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        print(f"Connected to Qdrant at {qdrant_url}")
        # Ensure payload index exists for category filtering
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="category_id",
                field_schema="integer",
            )
            print("Created payload index on category_id")
        except Exception:
            pass  # Already exists
    else:
        print("WARNING: QDRANT_URL / QDRANT_API_KEY not set, search disabled")

    # Process sample images through YOLOS+SAM2 on startup
    _cache_dir = Path(__file__).parent / "static" / "sample_cache"
    _manifest_path = _cache_dir / "manifest.json"
    if _manifest_path.exists():
        with open(_manifest_path) as f:
            manifest = json.load(f)
        loaded = 0
        for filename, meta in manifest.items():
            h = meta["hash"]
            ext = meta.get("ext", ".jpg")
            img_path = _cache_dir / f"{h}{ext}"
            if img_path.exists() and h not in seg_cache:
                img = Image.open(img_path).convert("RGB")
                print(f"Processing sample {filename} with YOLOS+SAM2...")
                instances = detect_and_segment(img)
                seg_map = build_seg_map_from_instances(instances, img.size[0], img.size[1])
                seg_cache[h] = {
                    "seg_map": seg_map,
                    "instances": instances,
                    "orig_w": meta["orig_w"],
                    "orig_h": meta["orig_h"],
                    "image": img,
                }
                detected = [(inst["category_name"], f"{inst['score']:.2f}") for inst in instances]
                print(f"  detected: {detected}")
                loaded += 1
        print(f"Processed {loaded} sample(s) with YOLOS+SAM2")



def _build_labels_and_clothing_ids(entry: dict) -> tuple[dict, list[int]]:
    """Build labels dict and clothing IDs list from a cache entry."""
    labels = {"0": "Background"}
    clothing_ids = []
    for inst in entry.get("instances", []):
        iid = inst["instance_id"]
        labels[str(iid)] = inst["category_name"]
        clothing_ids.append(iid)
    return labels, clothing_ids


def _encode_segmap(seg_map: np.ndarray) -> str:
    """Encode seg_map as a grayscale PNG (pixel value = category ID)."""
    img = Image.fromarray(seg_map, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/encode")
async def encode_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    h = image_hash(data)

    if h not in seg_cache:
        print(f"[encode] {img.size[0]}x{img.size[1]} running YOLOS+SAM2...")
        instances = detect_and_segment(img)
        seg_map = build_seg_map_from_instances(instances, img.size[0], img.size[1])

        seg_cache[h] = {
            "seg_map": seg_map,
            "instances": instances,
            "orig_w": img.size[0],
            "orig_h": img.size[1],
            "image": img,
        }

        detected = [(inst["category_name"], f"{inst['score']:.2f}") for inst in instances]
        print(f"[encode] detected: {detected}")

    entry = seg_cache[h]
    seg_b64 = _encode_segmap(entry["seg_map"])
    labels, clothing_ids = _build_labels_and_clothing_ids(entry)

    return {
        "hash": h,
        "width": entry["orig_w"],
        "height": entry["orig_h"],
        "segMap": seg_b64,
        "labels": labels,
        "clothingIds": clothing_ids,
    }


@app.get("/api/segmap/{hash}")
async def get_segmap(hash: str):
    """Return the pre-computed seg_map as a base64 grayscale PNG."""
    if hash not in seg_cache:
        return JSONResponse({"error": "Image not encoded"}, 400)
    entry = seg_cache[hash]
    seg_b64 = _encode_segmap(entry["seg_map"])
    labels, clothing_ids = _build_labels_and_clothing_ids(entry)
    return {
        "segMap": seg_b64,
        "width": entry["orig_w"],
        "height": entry["orig_h"],
        "labels": labels,
        "clothingIds": clothing_ids,
    }


@app.post("/api/search")
async def search_item(body: dict):
    """Search for similar products based on clicked clothing item."""
    h = body["hash"]
    instance_id = int(body["categoryId"])  # seg_map pixel value (now instance ID)
    limit = int(body.get("limit", 100))
    offset = int(body.get("offset", 0))

    if h not in seg_cache:
        return JSONResponse({"error": "Image not encoded"}, 400)

    entry = seg_cache[h]
    seg_map = entry["seg_map"]
    img = entry["image"]

    # Find the instance by its ID
    instance = None
    for inst in entry.get("instances", []):
        if inst.get("instance_id") == instance_id:
            instance = inst
            break
    if not instance:
        return JSONResponse({"error": "Instance not found"}, 400)

    mask = (seg_map == instance_id).astype(np.uint8) * 255
    color = dominant_color_name(img, mask)

    actual_cat_id = instance["category_id"]
    search_term = YOLOS_SEARCH_NAMES.get(actual_cat_id, YOLOS_LABELS.get(actual_cat_id, "clothing"))
    fashionpedia_ids = YOLOS_TO_FASHIONPEDIA.get(actual_cat_id)
    cat_name = YOLOS_LABELS.get(actual_cat_id, "unknown")

    query = f"{color} {search_term}".strip()
    print(f"[search] category={cat_name} color={color} query=\"{query}\" filter={fashionpedia_ids}")

    products = search_qdrant(query, limit=limit, offset=offset, category_ids=fashionpedia_ids)

    return {
        "query": query,
        "products": products,
    }


@app.get("/api/similar/{point_id}")
async def find_similar(
    point_id: int,
    limit: int = Query(default=100),
    offset: int = Query(default=0),
):
    """Find similar products by Qdrant point ID."""
    if not qdrant:
        return JSONResponse({"error": "Search not available"}, 503)

    try:
        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=point_id,
            limit=limit,
            offset=offset,
        )
        products = [_point_to_product(point) for point in results.points]
        return {"query": f"similar to #{point_id}", "products": products}
    except Exception as e:
        print(f"[similar] error: {e}")
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/sample-hashes")
async def sample_hashes():
    """Return pre-computed hashes for sample images."""
    _manifest_path = Path(__file__).parent / "static" / "sample_cache" / "manifest.json"
    if not _manifest_path.exists():
        return {}
    with open(_manifest_path) as f:
        manifest = json.load(f)
    # Map "/samples/{filename}" → hash
    return {f"/samples/{k}": v["hash"] for k, v in manifest.items()}


@app.get("/api/health")
async def health():
    return {"status": "ok", "searchEnabled": qdrant is not None and bool(GEMINI_API_KEY)}

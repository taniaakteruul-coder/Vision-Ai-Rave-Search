import os
import csv
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import models, transforms


DATA_DIR = "data"
ITEMS_DIR = os.path.join(DATA_DIR, "found_items")
CSV_FILE = os.path.join(DATA_DIR, "found_items_metadata.csv")

ART_DIR = "artifacts"
EMB_FILE = os.path.join(ART_DIR, "embeddings.npy")
META_FILE = os.path.join(ART_DIR, "metadata.json")
INFO_FILE = os.path.join(ART_DIR, "model_info.json")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ITEMS_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    # keep folders in git
    open(os.path.join(DATA_DIR, ".gitkeep"), "a").close()
    os.makedirs(ITEMS_DIR, exist_ok=True)
    open(os.path.join(ITEMS_DIR, ".gitkeep"), "a").close()


def init_metadata_csv():
    if os.path.exists(CSV_FILE):
        return
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "item_id",
            "item_name",
            "found_place",
            "date_found",
            "image_path",
            "finder_full_name",
            "finder_contact_number",
            "finder_person_type",
            "finder_student_id",
            "finder_purpose",
            "finder_purpose_other",
        ])


def normalize_text(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def normalize_phone(s: str) -> str:
    # simple: keep digits and optional leading +
    s = (s or "").strip()
    if s.startswith("+"):
        return "+" + "".join([c for c in s[1:] if c.isdigit()])
    return "".join([c for c in s if c.isdigit()])


def load_found_items() -> List[Dict]:
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def save_found_item(
    item_name: str,
    found_place: str,
    photo_bytes: Optional[bytes],
    photo_ext: Optional[str],
    finder_full_name: str,
    finder_contact: str,
    finder_person_type: str,
    finder_student_id: str,
    finder_purpose: str,
    finder_purpose_other: str,
) -> Dict:
    ensure_dirs()
    init_metadata_csv()

    item_id = str(uuid.uuid4())[:8]
    date_found = datetime.now().strftime("%Y-%m-%d %H:%M")

    image_path = ""
    if photo_bytes:
        ext = (photo_ext or "jpg").lower().replace(".", "")
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            ext = "jpg"
        filename = f"{item_id}.{ext}"
        image_path = os.path.join(ITEMS_DIR, filename).replace("\\", "/")
        with open(image_path, "wb") as f:
            f.write(photo_bytes)

    row = {
        "item_id": item_id,
        "item_name": normalize_text(item_name),
        "found_place": found_place.strip(),
        "date_found": date_found,
        "image_path": image_path,
        "finder_full_name": finder_full_name.strip(),
        "finder_contact_number": normalize_phone(finder_contact),
        "finder_person_type": finder_person_type.strip(),
        "finder_student_id": finder_student_id.strip(),
        "finder_purpose": finder_purpose.strip(),
        "finder_purpose_other": finder_purpose_other.strip(),
    }

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(row.values()))

    return row


def filter_items_text(items: List[Dict], item_name: str, place: str) -> List[Dict]:
    qn = normalize_text(item_name)
    qp = normalize_text(place)

    out = []
    for it in items:
        name_ok = qn in normalize_text(it.get("item_name", ""))
        place_ok = qp in normalize_text(it.get("found_place", ""))
        if name_ok and place_ok:
            out.append(it)
    return out


# -------------------------
# CV model: ResNet50 embeddings
# -------------------------
def get_model(device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    return model, preprocess, 2048, "resnet50_imagenet", device


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n


def load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Invalid image file: {path}") from e


@torch.no_grad()
def embed_image(model, preprocess, img: Image.Image, device: str) -> np.ndarray:
    x = preprocess(img).unsqueeze(0).to(device)
    v = model(x).detach().cpu().numpy().astype(np.float32)[0]
    v = l2_normalize(v)
    return v


def build_embedding_index():
    ensure_dirs()
    items = load_found_items()
    items_with_images = [it for it in items if it.get("image_path")]

    if not items_with_images:
        raise ValueError("No images found in the database to index.")

    model, preprocess, dim, name, device = get_model()

    embs = []
    meta = []

    for it in items_with_images:
        img = load_image(it["image_path"])
        v = embed_image(model, preprocess, img, device)
        embs.append(v)
        meta.append(it)

    embs = np.vstack(embs).astype(np.float32)
    embs = l2_normalize(embs)

    np.save(EMB_FILE, embs)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    info = {
        "model": name,
        "embedding_dim": dim,
        "count": len(meta),
        "built_at": datetime.now().isoformat(),
    }
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_index():
    if not (os.path.exists(EMB_FILE) and os.path.exists(META_FILE)):
        raise FileNotFoundError("Image index not found. Please build the index first.")
    embs = np.load(EMB_FILE).astype(np.float32)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embs, meta


def search_by_image(query_img: Image.Image, top_k: int = 5) -> List[Tuple[Dict, float]]:
    embs, meta = load_index()
    model, preprocess, _, _, device = get_model()

    q = embed_image(model, preprocess, query_img, device).astype(np.float32)
    q = l2_normalize(q)

    sims = embs @ q  # cosine similarity because vectors are normalized
    idxs = np.argsort(-sims)[:top_k]
    return [(meta[i], float(sims[i])) for i in idxs]

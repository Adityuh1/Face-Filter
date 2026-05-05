import os
# CRITICAL: Ensures DeepFace works with TensorFlow 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

import cv2 # to check if image is blurry or not feature

import chromadb
from chromadb.config import Settings
from deepface import DeepFace

# -----------------------------------------------------------------------------
# ChromaDB setup
# -----------------------------------------------------------------------------


@st.cache_resource
def get_chroma_client():
    """Returns a client with reset permissions enabled."""
    return chromadb.PersistentClient(
        path="chroma_db_data",
        # This is the "Key" that unlocks the reset() function
        settings=Settings(allow_reset=True)
    )
def _ensure_cosine_collection(client, name: str) -> None:
    """
    Embeddings + thresholds assume cosine distance. Collections created with
    default L2 space make similarity thresholds meaningless.
    """
    try:
        coll = client.get_collection(name=name)
        md = coll.metadata or {}
        if md.get("hnsw:space") != "cosine":
            client.delete_collection(name=name)
    except Exception:
        pass


@st.cache_resource
def get_collections(_schema_version: int = 3):
    """Get or create the two main collections (cosine space, multi-face rows). Bump _schema_version if Chroma layout changes."""
    client = get_chroma_client()
    _ensure_cosine_collection(client, "face_vectors")
    _ensure_cosine_collection(client, "scene_vectors")
    face_vectors = client.get_or_create_collection(
        name="face_vectors",
        metadata={"hnsw:space": "cosine"},
    )
    scene_vectors = client.get_or_create_collection(
        name="scene_vectors",
        metadata={"hnsw:space": "cosine"},
    )
    return face_vectors, scene_vectors

# -----------------------------------------------------------------------------
# Model loading (cached for performance)
# -----------------------------------------------------------------------------

@st.cache_resource
def get_clip_model() -> SentenceTransformer:
    """Load the CLIP model once and reuse."""
    return SentenceTransformer("clip-ViT-B-16")

# -----------------------------------------------------------------------------
# DeepFace logic
# -----------------------------------------------------------------------------

def _represent_as_list(reps: Any) -> List[dict]:
    if reps is None:
        return []
    if isinstance(reps, dict):
        return [reps]
    if isinstance(reps, list):
        if reps and isinstance(reps[0], list):
            return [x for group in reps for x in group if isinstance(x, dict)]
        return [x for x in reps if isinstance(x, dict)]
    return []


def _face_area(rep: dict) -> float:
    fa = rep.get("facial_area") or rep.get("face_area")
    if isinstance(fa, dict):
        return float(fa.get("w", 0) * fa.get("h", 0))
    if isinstance(fa, (int, float)):
        return float(fa)
    region = rep.get("region")
    if isinstance(region, dict):
        return float(region.get("w", 0) * region.get("h", 0))
    if isinstance(region, (list, tuple)) and len(region) >= 4:
        return float(region[2]) * float(region[3])
    return 0.0


def get_all_faces_data(
    image_path: str | Path,
    *,
    enforce_detection: bool = True,
    max_faces: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Extract every detected face: embedding + emotion (best-effort per face).
    Embeddings are L2-normalized for cosine distance in Chroma.
    """
    img_path = str(image_path)
    kwargs = dict(
        img_path=img_path,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=enforce_detection,
        align=True,
        l2_normalize=True,
    )
    if max_faces is not None:
        kwargs["max_faces"] = max_faces

    reps = DeepFace.represent(**kwargs)
    face_reps = _represent_as_list(reps)
    if not face_reps:
        return []

    analyses = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion"],
        detector_backend="retinaface",
        enforce_detection=enforce_detection,
    )
    analysis_list: List[dict] = []
    if isinstance(analyses, list):
        analysis_list = [a for a in analyses if isinstance(a, dict)]
    elif isinstance(analyses, dict):
        analysis_list = [analyses]

    out: List[Dict[str, Any]] = []
    for i, rep in enumerate(face_reps):
        emb = np.array(rep.get("embedding", []), dtype="float32")
        if emb.size == 0:
            continue
        emotion = "unknown"
        if i < len(analysis_list):
            emotion = str(analysis_list[i].get("dominant_emotion", "unknown"))
        elif analysis_list:
            emotion = str(analysis_list[0].get("dominant_emotion", "unknown"))
        out.append(
            {
                "embedding": emb,
                "emotion": emotion,
                "facial_area": _face_area(rep),
            }
        )
    return out


def get_face_data(image_path: str | Path) -> Dict[str, Any]:
    """Single primary face (largest detection) for legacy callers / query images."""
    faces = get_all_faces_data(image_path, enforce_detection=True)
    if not faces:
        faces = get_all_faces_data(image_path, enforce_detection=False)
    if not faces:
        return {"embedding": np.array([], dtype="float32"), "emotion": "unknown"}
    primary = max(faces, key=lambda f: f.get("facial_area", 0.0))
    return {"embedding": primary["embedding"], "emotion": primary["emotion"]}


def get_query_face_data(image_path: str | Path) -> Dict[str, Any]:
    """Reference search: prefer largest face; relax detection if needed."""
    faces = get_all_faces_data(image_path, enforce_detection=True, max_faces=12)
    if not faces:
        faces = get_all_faces_data(image_path, enforce_detection=False, max_faces=12)
    if not faces:
        return {"embedding": np.array([], dtype="float32"), "emotion": "unknown"}
    primary = max(faces, key=lambda f: f.get("facial_area", 0.0))
    return {"embedding": primary["embedding"], "emotion": primary["emotion"]}

# -----------------------------------------------------------------------------
# Indexer
# -----------------------------------------------------------------------------

def _file_row_key(file_path: str) -> str:
    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()


def index_image(image_path: str | Path) -> None:
    """
    Index one image: CLIP scene (one row per file) + every detected face
    (one row per face). Vector ids are derived from the resolved file path
    so re-indexing the same file replaces prior rows.
    """
    image_path = Path(image_path)
    file_path = str(image_path.resolve())
    row_key = _file_row_key(file_path)
    face_vectors, scene_vectors = get_collections()

    try:
        existing = face_vectors.get(where={"file_path": file_path}, include=["metadatas"])
        if existing.get("ids"):
            face_vectors.delete(ids=existing["ids"])
    except Exception:
        pass

    try:
        existing_scene = scene_vectors.get(where={"file_path": file_path}, include=["metadatas"])
        if existing_scene.get("ids"):
            scene_vectors.delete(ids=existing_scene["ids"])
    except Exception:
        pass

    face_rows = get_all_faces_data(file_path, enforce_detection=True)
    if not face_rows:
        face_rows = get_all_faces_data(file_path, enforce_detection=False)
    n_faces = len(face_rows)

    if n_faces:
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        for i, row in enumerate(face_rows):
            ids.append(f"{row_key}_f{i}")
            embeddings.append(row["embedding"].tolist())
            metadatas.append(
                {
                    "file_path": file_path,
                    "emotion": row["emotion"],
                    "face_index": int(i),
                    "faces_in_image": int(n_faces),
                }
            )
        face_vectors.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    model = get_clip_model()
    img = Image.open(file_path).convert("RGB")
    scene_emb = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]  # type: ignore
    scene_vectors.upsert(
        ids=[f"{row_key}_scene"],
        embeddings=[scene_emb.astype("float32").tolist()],
        metadatas=[{"file_path": file_path}],
    )

# -----------------------------------------------------------------------------
# Search: by face (With Threshold Logic)
# -----------------------------------------------------------------------------

def _parse_faces_in_image(meta: dict | None) -> int:
    if not isinstance(meta, dict):
        return 999
    raw = meta.get("faces_in_image", 1)
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


def search_by_face(
    query_image_path: str | Path,
    top_k: int = 10,
    threshold: float = 0.6,
) -> List[str]:
    """
    Cosine distance in Chroma: for L2-normalized vectors this matches
    d ≈ 1 - cos(theta). Keep matches with d < threshold; dedupe by image;
    order solo portraits before multi-face images.
    """
    face_vectors, _ = get_collections()
    if face_vectors.count() == 0:
        return []

    query_data = get_query_face_data(query_image_path)
    query_emb = query_data.get("embedding")
    if query_emb is None or getattr(query_emb, "size", 0) == 0:
        return []

    n_probe = min(face_vectors.count(), max(top_k * 8, 64))
    results = face_vectors.query(
        query_embeddings=[query_emb.tolist()],
        n_results=n_probe,
        include=["metadatas", "distances"],
    )

    if not results.get("metadatas") or not results.get("distances"):
        return []

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    best: Dict[str, Tuple[float, int]] = {}
    for meta, dist in zip(metadatas, distances):
        if meta is None or dist is None:
            continue
        if float(dist) >= threshold:
            continue
        path = meta.get("file_path")
        if not path:
            continue
        fin = _parse_faces_in_image(meta)
        prev = best.get(path)
        if prev is None or float(dist) < prev[0]:
            best[path] = (float(dist), fin)

    ranked = sorted(
        best.items(),
        key=lambda item: (0 if item[1][1] == 1 else 1, item[1][1], item[1][0], item[0].lower()),
    )
    out = [p for p, _ in ranked]
    return out[:top_k]

# -----------------------------------------------------------------------------
# Search: by text (CLIP)
# -----------------------------------------------------------------------------

def search_by_text(
    query_text: str,
    top_k: int = 24,
    *,
    max_distance: float | None = 0.48,
) -> List[str]:
    """
    CLIP text → image search with multi-query embedding (averaged prompts) and
    optional cosine-distance filtering. If the cutoff removes everything, falls
    back to the best matches by distance.
    """
    q = (query_text or "").strip()
    if not q:
        return []

    _, scene_vectors = get_collections()
    n_docs = scene_vectors.count()
    if n_docs == 0:
        return []

    model = get_clip_model()
    prompts = [
        q,
        f"a photograph of {q}",
        f"a photo of {q}",
        f"an image showing {q}",
    ]
    embs = model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
    text_emb = np.asarray(embs, dtype=np.float64).mean(axis=0)
    text_emb /= float(np.linalg.norm(text_emb) + 1e-12)
    text_emb = text_emb.astype("float32")

    n_probe = min(n_docs, max(top_k * 8, 80))
    results = scene_vectors.query(
        query_embeddings=[text_emb.tolist()],
        n_results=n_probe,
        include=["metadatas", "distances"],
    )

    if not results.get("metadatas") or not results.get("distances"):
        return []

    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ranked: List[Tuple[float, str]] = []
    for meta, dist in zip(metas, dists):
        if meta is None or dist is None:
            continue
        path = meta.get("file_path")
        if not path:
            continue
        ranked.append((float(dist), str(path)))

    ranked.sort(key=lambda x: x[0])

    filtered = [(d, p) for d, p in ranked if max_distance is None or d <= max_distance]
    chosen = filtered if filtered else ranked

    out: List[str] = []
    seen: set[str] = set()
    for _, p in chosen:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= top_k:
            break
    return out

# -----------------------------------------------------------------------------
# Clustering (DBSCAN)
# -----------------------------------------------------------------------------

def cluster_faces(eps: float = 0.45, min_samples: int = 2) -> Dict[str, List[str]]:
    face_vectors, _ = get_collections()
    if face_vectors.count() == 0:
        return {}

    data = face_vectors.get(include=["embeddings", "metadatas"])
    raw_embeddings = data.get("embeddings")

    if raw_embeddings is None or len(raw_embeddings) == 0:
        return {}
    
    embeddings = np.array(raw_embeddings, dtype="float32")
    metadatas = data.get("metadatas") or []

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    clusters: Dict[str, List[Tuple[str, int]]] = {}
    for label, meta in zip(labels, metadatas):
        if not isinstance(meta, dict):
            continue
        path = meta.get("file_path")
        if not path:
            continue
        person_label = f"Person {label}" if label != -1 else "Unknown"
        fin = _parse_faces_in_image(meta)
        clusters.setdefault(person_label, []).append((path, fin))

    out: Dict[str, List[str]] = {}
    for person_label, pairs in clusters.items():
        seen: Dict[str, int] = {}
        for path, fin in pairs:
            if path not in seen:
                seen[path] = fin
            else:
                seen[path] = min(seen[path], fin)
        paths = sorted(
            seen.items(),
            key=lambda item: (0 if item[1] == 1 else 1, item[1], item[0].lower()),
        )
        out[person_label] = [p for p, _ in paths]

    return out


def count_estimated_people() -> int:
    """
    Approximate distinct people from face embeddings. Uses several DBSCAN radii
    and takes the maximum cluster count so merged groups (too few clusters at
    one eps) do not under-report as badly as a single loose clustering.
    """
    face_vectors, _ = get_collections()
    c = face_vectors.count()
    if c == 0:
        return 0
    if c == 1:
        return 1

    data = face_vectors.get(include=["embeddings", "metadatas"])
    raw_embeddings = data.get("embeddings")
    if raw_embeddings is None or len(raw_embeddings) == 0:
        return 0

    embeddings = np.array(raw_embeddings, dtype="float32")
    best = 0
    for eps in (0.26, 0.30, 0.34, 0.38, 0.42, 0.46):
        labels = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(embeddings)
        k = len({int(l) for l in labels if int(l) >= 0})
        best = max(best, k)

    strict_labels = DBSCAN(eps=0.26, min_samples=2, metric="cosine").fit_predict(embeddings)
    noise = int((strict_labels == -1).sum())
    n_clusters_strict = len({int(l) for l in strict_labels if int(l) >= 0})
    # Tight clustering: each cluster is at least one identity; each noise face is
    # often a separate person not yet linked to a pair.
    combined = n_clusters_strict + noise
    return min(c, max(best, combined, 1))


def reset_database():
    """Wipes all data and clears the app's 'memory' (cache)."""
    client = get_chroma_client()
    try:
        # 1. The physical wipe
        client.reset() 
        
        # 2. The mental wipe (Cache Invalidation)
        # This tells Streamlit: "Throw away every @st.cache_resource"
        st.cache_resource.clear() 
        
        st.success("Database wiped! Collections will be recreated on next use.")
    except Exception as e:
        st.error(f"Reset failed: {e}")
        
# to check if image is blurry or not        
def is_image_blurry(image_path: str, threshold: float = 100.0) -> bool:
    """
    Checks if an image is blurry using the Laplacian method.
    Higher threshold = stricter check (e.g., 150).
    Lower threshold = more relaxed (e.g., 50).
    """
    image = cv2.imread(image_path)
    if image is None:
        return True, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If variance is below the threshold, it's blurry
    return variance < threshold, variance
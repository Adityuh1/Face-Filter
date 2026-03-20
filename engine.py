import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

import chromadb
from deepface import DeepFace


# -----------------------------------------------------------------------------
# ChromaDB setup
# -----------------------------------------------------------------------------


@st.cache_resource
def get_chroma_client() -> chromadb.Client:
    """Return a persistent Chroma client stored in ./chroma_db_data."""
    return chromadb.PersistentClient(path="chroma_db_data")


@st.cache_resource
def get_collections():
    """
    Get or create the two main collections:
    - facnahh e_vectors: face embeddings + emotion + file_path
    - scene_vectors: CLIP embeddings + file_path
    """
    client = get_chroma_client()
    face_vectors = client.get_or_create_collection(name="face_vectors")
    scene_vectors = client.get_or_create_collection(name="scene_vectors")
    return face_vectors, scene_vectors


# -----------------------------------------------------------------------------
# Model loading (cached for performance)
# -----------------------------------------------------------------------------


@st.cache_resource
def get_clip_model() -> SentenceTransformer:
    """Load the CLIP model once and reuse."""
    return SentenceTransformer("clip-ViT-B-32")


# -----------------------------------------------------------------------------
# DeepFace logic
# -----------------------------------------------------------------------------


def get_face_data(image_path: str | Path) -> Dict[str, Any]:
    """
    Compute a face embedding and dominant emotion for a given image.
    Uses Facenet512 and OpenCV backend to avoid dlib.
    """
    img_path = str(image_path)

    # DeepFace.represent returns a list of representations; take the first
    reps = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=False,
    )
    rep = reps[0] if isinstance(reps, list) else reps
    embedding = np.array(rep["embedding"], dtype="float32")

    # Emotion analysis
    analysis = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=False,
    )
    if isinstance(analysis, list):
        analysis = analysis[0]

    emotion = None
    if isinstance(analysis, dict):
        emotion = analysis.get("dominant_emotion")
        if not emotion and "emotion" in analysis and isinstance(analysis["emotion"], dict):
            # Some DeepFace versions keep a nested 'emotion' dict
            emotion = analysis["emotion"].get("dominant")

    return {
        "embedding": embedding,
        "emotion": emotion or "unknown",
    }


# -----------------------------------------------------------------------------
# CLIP logic
# -----------------------------------------------------------------------------


def _load_image(image_path: str | Path) -> Image.Image:
    """Load an image with Pillow and convert to RGB."""
    img = Image.open(str(image_path))
    return img.convert("RGB")


def get_scene_embedding(image_path: str | Path) -> np.ndarray:
    """
    Get a CLIP embedding for the whole image using SentenceTransformer CLIP model.
    """
    model = get_clip_model()
    img = _load_image(image_path)
    emb = model.encode(
        [img],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb.astype("float32")


# -----------------------------------------------------------------------------
# Indexer
# -----------------------------------------------------------------------------


def index_image(image_path: str | Path) -> None:
    """
    Index a single image into both Chroma collections:
    - Face embedding + emotion into 'face_vectors'
    - CLIP scene embedding into 'scene_vectors'
    """
    image_path = Path(image_path)
    file_path = str(image_path.resolve())
    img_id = image_path.name

    face_vectors, scene_vectors = get_collections()

    # Face data
    face_data = get_face_data(file_path)
    face_vectors.upsert(
        ids=[img_id],
        embeddings=[face_data["embedding"].tolist()],
        metadatas=[
            {
                "file_path": file_path,
                "emotion": face_data["emotion"],
            }
        ],
    )

    # Scene (CLIP) data
    scene_embedding = get_scene_embedding(file_path)
    scene_vectors.upsert(
        ids=[img_id],
        embeddings=[scene_embedding.tolist()],
        metadatas=[
            {
                "file_path": file_path,
            }
        ],
    )


# -----------------------------------------------------------------------------
# Search: by face
# -----------------------------------------------------------------------------


def search_by_face(query_image_path: str | Path, top_k: int = 5) -> List[str]:
    """
    Given a query face image, return paths of the top-k most similar faces
    from the 'face_vectors' collection.
    """
    face_vectors, _ = get_collections()
    if face_vectors.count() == 0:
        return []

    query_data = get_face_data(query_image_path)
    query_emb = query_data["embedding"].tolist()

    results = face_vectors.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas", "distances", "ids"],
    )

    metadatas = results.get("metadatas") or []
    if not metadatas:
        return []

    # results["metadatas"] is a list per query; we only have one query
    hits = metadatas[0]
    paths: List[str] = []
    for m in hits:
        path = m.get("file_path")
        if path:
            paths.append(path)

    return paths


# -----------------------------------------------------------------------------
# Search: by text (CLIP)
# -----------------------------------------------------------------------------


def search_by_text(query_text: str, top_k: int = 20) -> List[str]:
    """
    Given a natural language query, return paths of the most relevant images
    using the CLIP scene embeddings.
    """
    _, scene_vectors = get_collections()
    if scene_vectors.count() == 0:
        return []

    model = get_clip_model()
    text_emb = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype("float32")

    results = scene_vectors.query(
        query_embeddings=[text_emb.tolist()],
        n_results=top_k,
        include=["metadatas", "distances", "ids"],
    )

    metadatas = results.get("metadatas") or []
    if not metadatas:
        return []

    hits = metadatas[0]
    paths: List[str] = []
    for m in hits:
        path = m.get("file_path")
        if path:
            paths.append(path)

    return paths


# -----------------------------------------------------------------------------
# Emotion filter helper
# -----------------------------------------------------------------------------


def filter_by_emotion_from_db(emotion: str) -> List[str]:
    """
    Return all image paths from 'face_vectors' whose stored dominant emotion matches.
    Case-insensitive comparison.
    """
    face_vectors, _ = get_collections()
    if face_vectors.count() == 0:
        return []

    data = face_vectors.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    matches: List[str] = []
    target = emotion.lower()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        emo = str(meta.get("emotion", "")).lower()
        if emo == target:
            path = meta.get("file_path")
            if path:
                matches.append(path)

    return matches


# -----------------------------------------------------------------------------
# Clustering (DBSCAN) - group photos by person
# -----------------------------------------------------------------------------


def cluster_faces(eps: float = 0.6, min_samples: int = 2) -> Dict[str, List[str]]:
    """
    Cluster all face embeddings using DBSCAN and return a mapping:
    {
        "Person 0": [file_path1, file_path2, ...],
        "Person 1": [...],
        "Unknown":  [...],  # noise / unclustered
    }
    """
    face_vectors, _ = get_collections()
    if face_vectors.count() == 0:
        return {}

    data = face_vectors.get(include=["embeddings", "metadatas", "ids"])
    embeddings = np.array(data.get("embeddings") or [], dtype="float32")
    metadatas = data.get("metadatas") or []

    if len(embeddings) == 0:
        return {}

    # DBSCAN clustering over the embedding space
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    clusters: Dict[str, List[str]] = {}
    for label, meta in zip(labels, metadatas):
        if not isinstance(meta, dict):
            continue
        path = meta.get("file_path")
        if not path:
            continue

        if label == -1:
            person_label = "Unknown"
        else:
            person_label = f"Person {label}"

        clusters.setdefault(person_label, []).append(path)

    return clusters


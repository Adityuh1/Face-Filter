import os
# CRITICAL: DeepFace models (FaceNet, VGG-Face) were built on Keras 2. 
# TensorFlow 2.16+ uses Keras 3 by default. This line forces compatibility.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import streamlit as st

# The "Brain" Libraries
import chromadb
from deepface import DeepFace
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from PIL import Image

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="chroma_db_dta")

@st.cache_resource
def get_collections():
    client = get_chroma_client()
    
    face_vectors = client.get_or_create_collection(name="face_vectors")
    scene_vectors = client.get_or_create_collection(name="scene_vectors")
    return face_vectors , scene_vectors

@st.cache_resource
def get_clip_model():
    return SentenceTransformer("clip-ViT-B-32")

def get_face_data(image_path: str | Path) -> Dict[str , Any]:
    reps = DeepFace.represent(
        img_path = str(image_path),
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=False,
    )
    
    rep = rep[0] if isinstance(reps , list) else reps
    
    analysis = DeepFace.analyze(
        img_path= str(image_path),
        actions= ["emotions"],
        detector_backend="opencv",
        enforce_detection=False,
    )
    emotion = analysis[0]["dominant_emotion"] if isinstance(analysis , list) else analysis["dominant_emotion"]
    
    return{
        "embedding" : np.array(rep["embedding"],dtype="float32"),
        "emotion" : emotion
    }
    
def get_scene_embedding(image_path: str | Path):
    model = get_clip_model()
    img = Image.open(str(image_path)).convert("RGB")
    
    return model.encode([img], convert_to_numpy=True)[0]

# This function is the "Librarian." It takes a photo and puts it on the right shelf in ChromaDB.
def index_image(image_path : str | Path):
    file_path = str(Path(image_path).resolve())
    img_id = Path(image_path).name
    
    face_vectors , scene_vectors = get_collections()
    
    #1.Get the data
    face_data = get_face_data(file_path)
    scene_emb = get_scene_embedding(file_path)
    
    #2. Save the Face collection(Biometrics + Emotions)
    face_vectors.upsert(
        ids=[img_id],
        embeddings=[face_data["embeddings"].tolist()],
        metadatas=[{"file_path": file_path , "emotion" : face_data["emotion"]}]
    )
    
    # 3. Save to Scene collection (CLIP search)
    scene_vectors.upsert(
        ids=[img_id],
        embeddings=[scene_emb.tolist()],
        metadatas=[{"file_path": file_path}]
    )
    
    
def search_by_face(query_image_path: str | Path, top_k: int = 5) -> List[str]:
    face_vectors , _ = get_collections()
    if face_vectors.count() == 0:
        return []
    
    # 1. Convert the new "Search" photo into a vector
    query_data = get_face_data(query_image_path)
    query_emb = query_data["embedding"].tolist()
    
    #2. Ask chromaDb for closest matches
    results = face_vectors.query(
        query_embeddings=[query_emb],
        n_results = top_k , 
        include=["metadatas"]
    )
    return [m["file_path"] for m in results["metadatas"][0]]

def search_by_text(query_text : str , top_k: int = 10) -> List[str]:
    
    _ , scene_vectors = get_collections()
    model = get_clip_model()
    
    #1. Convert 'Text' into a 'Vector'
    text_emb = model.encode([query_text])[0].tolist()
    
    #2. Query the scene collection
    results = scene_vectors.query(
        query_embeddings=[text_emb],
        n_results=top_k,
        include=["metadatas"]
    )
    return[m["file_path"] for m in results["metadatas"][0]]

def cluster_faces(eps: float = 0.5, min_samples: int = 2) -> Dict[str, List[str]]:
    """Groups all indexed faces into 'People' clusters automatically."""
    face_vectors, _ = get_collections()
    
    # 1. Pull all data from ChromaDB
    data = face_vectors.get(include=["embeddings", "metadatas"])
    embeddings = np.array(data["embeddings"], dtype="float32")
    metadatas = data["metadatas"]

    if len(embeddings) == 0:
        return {}

    # 2. Run DBSCAN (Density-Based Spatial Clustering)
    # metric="cosine" is vital for face recognition accuracy
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = model.labels_

    # 3. Organize the results into a dictionary
    clusters = {}
    for label, meta in zip(labels, metadatas):
        person_name = f"Person {label}" if label != -1 else "Unknown/Noise"
        clusters.setdefault(person_name, []).append(meta["file_path"])
        
    return clusters
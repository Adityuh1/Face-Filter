"""
Smart AI Photo Gallery - Streamlit frontend.
Navigation: Home, Upload/Index, Face Filter, Smart Search.
Placeholders for engine.py integration.
"""
import engine
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Smart AI Photo Gallery",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Placeholders for engine.py (to be implemented)
# ---------------------------------------------------------------------------


def get_total_photos() -> int:
    """Gets the count of images stored in the Scene collection."""
    _, scene_vectors = engine.get_collections()
    return scene_vectors.count()

def get_people_count() -> int:
    """Uses the clustering logic to count unique individuals."""
    clusters = engine.cluster_faces(eps=0.45)
    # We subtract 1 if 'Unknown' exists to get an accurate 'People' count
    count = len(clusters)
    if "Unknown" in clusters:
        count -= 1
    return max(0, count)

def get_recent_photos(limit: int = 4) -> list:
    """Retrieves the last few photos added to the gallery."""
    _, scene_vectors = engine.get_collections()
    if scene_vectors.count() == 0:
        return []
    # Fetching all to show a 'preview' on the home page
    results = scene_vectors.get(limit=limit, include=["metadatas"])
    return [m["file_path"] for m in results["metadatas"]]

def index_uploaded_files(uploaded_files: list) -> None:
    # 1. Create a local folder to store these images permanently
    os.makedirs("gallery_photos", exist_ok=True)
    
    for uploaded_file in uploaded_files:
        save_path = os.path.join("gallery_photos" , uploaded_file.name)
        with open(save_path , "wb") as f:
            f.write(uploaded_file.getbuffer())
            
            engine.index_image(save_path)
            
def find_faces_by_reference(reference_image) -> list:
    """Find gallery photos matching the reference face. Replace with engine call."""
    return []


def clip_search(query: str, top_k: int = 20) -> list:
    """Search gallery by text using CLIP. Replace with engine call."""
    return []


def filter_by_emotion(emotion: str) -> list:
    """Filter gallery by emotion. Replace with engine call."""
    return []


def get_clusters() -> dict:
    """Get photo clusters by person (DBSCAN). Replace with engine call."""
    return {}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🖼️ Smart AI Gallery")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Upload / Index", "Face Filter", "Smart Search"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("College Project • AI-powered photo discovery")

# ---------------------------------------------------------------------------
# Home Page
# ---------------------------------------------------------------------------

if page == "Home":
    st.title("Smart AI Photo Gallery")
    st.markdown("Upload photos, find faces, and search by mood or natural language.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        total = get_total_photos()
        st.metric("Total Photos", total)
    with col2:
        people = get_people_count()
        st.metric("People Found", people)

    st.info("Use **Upload / Index** to add photos, **Face Filter** to find a face, and **Smart Search** for text and emotion filters.")

# ---------------------------------------------------------------------------
# Upload / Index Page
# ---------------------------------------------------------------------------

elif page == "Upload / Index":
    st.title("Upload & Index")
    st.markdown("Upload multiple photos to build your gallery. They will be indexed for face recognition, emotions, and search.")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Choose gallery images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Select one or more images to add to the gallery.",
    )

    if uploaded:
        st.success(f"{len(uploaded)} file(s) selected.")
        if st.button("Index photos"):
            with st.spinner("Indexing…"):
                index_uploaded_files(uploaded)
            st.success("Indexing complete. Check Home for updated stats.")
    else:
        st.caption("No files selected yet.")

# ---------------------------------------------------------------------------
# Face Filter Page
# ---------------------------------------------------------------------------

elif page == "Face Filter":
    st.title("Face Filter")
    st.markdown("Upload a reference photo to find all gallery images containing the same person.")
    st.markdown("---")

    ref = st.file_uploader(
        "Search face (reference photo)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="face_ref",
    )

    if ref is not None:
        st.image(ref, caption="Reference face", use_container_width=True)
        if st.button("Find matching faces"):
            with st.spinner("Searching gallery…"):
                matches = find_faces_by_reference(ref)
            if matches:
                st.subheader("Matching photos")
                n = len(matches)
                cols = st.columns(min(n, 4))
                for i, item in enumerate(matches):
                    # item can be path or (path, score); placeholder assumes path-like
                    path = item if isinstance(item, (str, Path)) else item[0]
                    with cols[i % len(cols)]:
                        st.image(path, use_container_width=True)
            else:
                st.warning("No matching faces found in the gallery.")
    else:
        st.caption("Upload a reference face to start.")

# ---------------------------------------------------------------------------
# Smart Search Page (tabs: Emotion, Clustering, Natural Language)
# ---------------------------------------------------------------------------

elif page == "Smart Search":
    st.title("Smart Search")
    st.markdown("Filter by emotion, browse clusters by person, or search with natural language.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Emotion Filter", "Clustering by Person", "Natural Language Search"])

    with tab1:
        st.subheader("Filter by mood")
        emotion = st.selectbox(
            "Emotion",
            ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"],
            key="emotion",
        )
        if st.button("Apply emotion filter", key="btn_emotion"):
            results = filter_by_emotion(emotion)
            if results:
                st.write(f"Found {len(results)} photo(s).")
                cols = st.columns(min(len(results), 4))
                for i, path in enumerate(results):
                    with cols[i % len(cols)]:
                        st.image(path, use_container_width=True)
            else:
                st.info("No photos match this emotion.")

    with tab2:
        st.subheader("Photos grouped by person")
        if st.button("Load clusters", key="btn_clusters"):
            clusters = get_clusters()
            if clusters:
                for label, paths in clusters.items():
                    st.write(f"**Person {label}** — {len(paths)} photo(s)")
                    cols = st.columns(min(len(paths), 4))
                    for i, path in enumerate(paths):
                        with cols[i % len(cols)]:
                            st.image(path, use_container_width=True)
            else:
                st.info("No clusters yet. Upload and index photos first.")

    with tab3:
        st.subheader("Search with text (CLIP)")
        query = st.text_input("Describe what you're looking for", placeholder="e.g. sunset, person eating")
        top_k = st.slider("Max results", 5, 50, 20, key="top_k")
        if st.button("Search", key="btn_clip"):
            if query.strip():
                with st.spinner("Searching…"):
                    results = clip_search(query.strip(), top_k=top_k)
                if results:
                    st.write(f"Top {len(results)} result(s).")
                    cols = st.columns(min(len(results), 4))
                    for i, item in enumerate(results):
                        path = item if isinstance(item, (str, Path)) else item[0]
                        with cols[i % len(cols)]:
                            st.image(path, use_container_width=True)
                else:
                    st.warning("No results. Try different keywords or add more photos.")
            else:
                st.warning("Enter a search query.")

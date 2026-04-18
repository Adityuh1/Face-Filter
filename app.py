"""
Smart AI Photo Gallery - Streamlit frontend.
Navigation: Home, Upload/Index, Face Filter, Text Search.
Placeholders for engine.py integration.
"""
import engine
import io
import os
import zipfile
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Safely gets the count, returning 0 if the collection is missing."""
    try:
        _, scene_vectors = engine.get_collections()
        return scene_vectors.count()
    except Exception:
        return 0

def get_people_count() -> int:
    """Approximate distinct people (face embeddings + DBSCAN at several scales)."""
    try:
        return engine.count_estimated_people()
    except Exception:
        return 0

def get_all_gallery_paths() -> list[str]:
    """All indexed image paths from the scene collection (deduped, sorted)."""
    try:
        _, scene_vectors = engine.get_collections()
        if scene_vectors.count() == 0:
            return []
        data = scene_vectors.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        seen: set[str] = set()
        out: list[str] = []
        for m in metas:
            if not isinstance(m, dict):
                continue
            p = m.get("file_path")
            if p and p not in seen:
                seen.add(p)
                out.append(str(p))
        return sorted(out, key=lambda x: x.lower())
    except Exception:
        return []


def build_image_paths_zip(paths: list[str]) -> bytes:
    """Pack existing files into a ZIP (streaming in memory). Skips missing paths."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        n = 0
        for p in paths:
            if not p or not os.path.isfile(p):
                continue
            n += 1
            arcname = f"{n:04d}_{Path(p).name}"
            zf.write(p, arcname=arcname)
    return buf.getvalue()

def index_uploaded_files(uploaded_files: list) -> None:
    """Save files on the main thread, then index up to 4 images at a time in a thread pool."""
    os.makedirs("gallery_photos", exist_ok=True)

    skip_messages: list[str] = []
    paths_to_index: list[str] = []

    for file in uploaded_files:
        save_path = os.path.join("gallery_photos", file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        is_blurry, score = engine.is_image_blurry(save_path)
        if is_blurry:
            skip_messages.append(
                f"⏩ Skipped '{file.name}': Image too blurry (Score: {score:.1f})"
            )
            os.remove(save_path)
            continue
        paths_to_index.append(save_path)

    for msg in skip_messages:
        st.warning(msg)

    if not paths_to_index:
        return

    def _index_one(path: str) -> tuple[str, str | None]:
        try:
            engine.index_image(path)
            return path, None
        except Exception as exc:
            return path, str(exc)

    total = len(paths_to_index)
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_index_one, p) for p in paths_to_index]
        done = 0
        for fut in as_completed(futures):
            path, err = fut.result()
            done += 1
            progress.progress(min(done / total, 1.0))
            status.text(f"Indexed {done} of {total}…")
            if err:
                st.warning(f"Indexing failed for `{path}`: {err}")

    progress.empty()
    status.empty()

def find_faces_by_reference(reference_image, *, top_k: int = 100, threshold: float = 0.6) -> list:
    """Saves the search photo temporarily and runs cosine face search."""
    temp_path = "temp_search_face.jpg"
    with open(temp_path, "wb") as f:
        f.write(reference_image.getbuffer())
    return engine.search_by_face(temp_path, top_k=top_k, threshold=threshold)
    # temp = "temp_search.jpg"
    # with open(temp, "wb") as f: f.write(ref_file.getbuffer())
    # return engine.search_by_face(temp)


def clip_search(query: str, top_k: int = 24, max_distance: float | None = 0.48) -> list:
    """Search gallery by text using CLIP (multi-query + distance filter in engine)."""
    return engine.search_by_text(query, top_k=top_k, max_distance=max_distance)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🖼️ Smart AI Gallery")
st.sidebar.markdown("---")

# page = st.sidebar.radio(
#     "Navigate",
#     ["Home", "Upload / Index", "Face Filter", "Smart Search"],
#     label_visibility="collapsed",
# )
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Upload / Index", "Face Filter", "Text Search"],
    key="main_navigation",
)


if st.sidebar.button("🗑️ Reset Gallery"):
    engine.reset_database()
    # Also delete the physical photos from the folder
    if os.path.exists("gallery_photos"):
        for file in os.listdir("gallery_photos"):
            os.remove(os.path.join("gallery_photos", file))
    st.session_state.pop("face_filter_matches", None)
    st.rerun()

if page == "Home":
    st.title("Smart AI Photo Gallery")
    c1, c2 = st.columns(2)
    c1.metric("Total Photos", get_total_photos())
    c2.metric("People (est.)", get_people_count())
    st.caption(
        "People count is an estimate from face embeddings (several cluster scales + "
        "unpaired faces). It is closer to “how many identities might be here” than "
        "a perfect head count."
    )
    st.info("Use the sidebar to upload and search your photos!")

    gallery_paths = get_all_gallery_paths()
    st.subheader("Your gallery")
    if not gallery_paths:
        st.caption("No indexed photos yet. Upload images from **Upload / Index**.")
    else:
        cols = st.columns(4)
        for i, p in enumerate(gallery_paths):
            cols[i % 4].image(p, use_container_width=True)

elif page == "Upload / Index":
    st.title("Upload & Index")
    st.caption(
        "After updating the engine, use **Reset Gallery** once, then re-index so "
        "faces use cosine distance and multi-face rows are stored correctly."
    )
    uploaded = st.file_uploader("Choose photos", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded and st.button("Index photos"):
        with st.spinner("AI is analyzing your photos..."):
            index_uploaded_files(uploaded)
        st.success("Indexing complete!")

elif page == "Face Filter":
    st.title("Face Filter")
    thr = st.slider("Match strictness (cosine distance, lower = stricter)", 0.35, 0.85, 0.60, 0.01)
    ref = st.file_uploader("Upload reference face", type=["jpg", "png"])
    if ref and st.button("Find matches"):
        matches = find_faces_by_reference(ref, top_k=100, threshold=float(thr))
        st.session_state["face_filter_matches"] = matches if matches else []
        if not matches:
            st.warning("No matches found.")

    match_paths: list[str] = st.session_state.get("face_filter_matches") or []
    if match_paths:
        zip_bytes = build_image_paths_zip(match_paths)
        if zip_bytes:
            st.download_button(
                label="Download all matches (ZIP)",
                data=zip_bytes,
                file_name="face_search_matches.zip",
                mime="application/zip",
                key="face_match_zip_download",
            )
        cols = st.columns(4)
        for i, p in enumerate(match_paths):
            cols[i % 4].image(p, use_container_width=True)

elif page == "Text Search":
    st.title("Search by text")
    st.caption(
        "Describe objects, colors, places, or activities. The engine averages several "
        "CLIP prompts and ranks images by cosine similarity, then drops weak matches."
    )
    query = st.text_input(
        "Search for…",
        placeholder="e.g. birthday cake, sunset at the beach, person in red shirt",
    )
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Max images to show", 8, 48, 24, 4)
    with c2:
        cutoff = st.slider(
            "Relevance cutoff (cosine distance; lower = stricter)",
            0.28,
            0.72,
            0.48,
            0.02,
        )
    if st.button("Search", type="primary"):
        q = (query or "").strip()
        if not q:
            st.warning("Enter a search phrase.")
        else:
            with st.spinner("Searching…"):
                paths = clip_search(q, top_k=int(top_k), max_distance=float(cutoff))
            if paths:
                cols = st.columns(4)
                for i, p in enumerate(paths):
                    cols[i % 4].image(p, use_container_width=True)
            else:
                st.warning(
                    "No images passed the relevance cutoff. Try a shorter phrase, "
                    "different wording, or raise the cutoff slightly."
                )
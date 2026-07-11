# Smart AI Photo Gallery

A local-first photo gallery that uses face recognition and natural language search to help you find photos — without uploading anything to the cloud.

Upload your photos once, and you can search them by describing what's in them ("sunset at the beach", "birthday cake") or by pointing to a face and finding every other photo that person appears in.

---

## What it does

**Face Filter** — Upload a reference photo of someone, and the app will scan your entire gallery for other photos containing that person. It uses facial embeddings (512-dimensional vectors from FaceNet) and cosine similarity to rank matches. You can tune how strict the matching is via a slider, and download all matches as a ZIP file.

**Text Search** — Describe what you're looking for in plain English. Under the hood, CLIP converts both your text and the gallery images into the same embedding space and finds the closest matches. It averages multiple prompt variations ("a photo of X", "an image showing X") to make searches more robust.

**Blur Detection** — Photos are checked for sharpness before indexing using the Laplacian variance method. Blurry images get rejected automatically so they don't pollute your search results.

**People Estimation** — The home screen shows an estimate of how many distinct people are in your gallery, computed by clustering face embeddings with DBSCAN at multiple epsilon values and taking the best result.

---

## Tech Stack

| Component | What it's doing |
|---|---|
| **Streamlit** | Web UI and session state management |
| **DeepFace + FaceNet512** | Face detection and 512-dim embedding extraction |
| **RetinaFace** | Face detector backend (used by DeepFace) |
| **CLIP (ViT-B/16)** | Image and text embedding for semantic search |
| **ChromaDB** | Local vector database for storing and querying embeddings |
| **DBSCAN (sklearn)** | Unsupervised clustering for people estimation |
| **OpenCV** | Blur detection via Laplacian variance |
| **Pillow** | Image loading and preprocessing |

---

## Project Structure

```
faceFilter/
├── app.py          # Streamlit frontend — pages, UI logic, file uploads
├── engine.py       # Core AI engine — indexing, search, clustering
├── gallery_photos/ # Uploaded and indexed images stored here
├── chroma_db_data/ # ChromaDB persistent storage (vector embeddings)
└── .env            # Environment variables (not committed)
```

`engine.py` and `app.py` are intentionally kept separate. The engine has no UI logic — it just takes file paths and returns results. This makes it easy to swap or test the backend independently.

---

## How the Search Works

### Face Search

1. A reference image is uploaded and its largest detected face is extracted.
2. FaceNet512 produces an L2-normalized 512-dim embedding for that face.
3. ChromaDB queries the `face_vectors` collection using cosine distance.
4. Results below the user-defined threshold are kept, deduplicated by image path, and ranked — solo portraits come before group photos.

### Text Search

1. The query is expanded into 4 prompt variations.
2. CLIP encodes all 4 prompts and averages the resulting vectors.
3. The averaged embedding is compared against scene-level image embeddings stored in `scene_vectors`.
4. Results are sorted by cosine distance and filtered by a configurable cutoff.

### Indexing

Each image gets two things stored:
- **Face rows** — one row per detected face, with its embedding and dominant emotion.
- **Scene row** — one CLIP embedding representing the whole image.

Re-indexing the same file replaces existing rows (IDs are derived from a SHA-256 hash of the file path).

---

## Setup

### Prerequisites

- Python 3.10+
- A machine with enough RAM to load CLIP and FaceNet simultaneously (~3–4 GB)

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** DeepFace requires TensorFlow. Set the environment variable `TF_USE_LEGACY_KERAS=1` if you're on TensorFlow 2.16+ (the app sets this automatically).

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload / Index** — Go to the Upload page, select your photos (JPG or PNG), and click "Index photos". The AI will analyze each one in parallel (up to 4 at a time).
2. **Home** — See your full gallery and a count of total photos and estimated unique people.
3. **Face Filter** — Upload a photo of someone, adjust the match strictness, and hit "Find matches". Download results as a ZIP if needed.
4. **Text Search** — Type a description of what you're looking for. Adjust the relevance cutoff and number of results as needed.
5. **Reset Gallery** — The sidebar button wipes the vector database and clears physical files. Use this if you want to start fresh or after updating the engine.

---

## Notes

- All data stays local. No embeddings or photos are sent to external servers.
- ChromaDB collections are created with cosine distance (`hnsw:space: cosine`). If you have an older database using the default L2 space, the app will automatically recreate the collections with the correct metric.
- The people count on the home screen is an estimate. It's better understood as "how many distinct identities might be here" rather than a precise head count.
- Models are loaded once and cached for the session using `@st.cache_resource`.

---

## License

This project is for personal and educational use.

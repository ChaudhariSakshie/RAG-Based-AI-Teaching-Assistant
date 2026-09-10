
import json
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(r"D:\AI_Projects(RAG)")

CHUNKS_DIR = BASE_DIR / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# ==========================================
# MODEL
# ==========================================

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ==========================================
# FIND ALL CHUNK FILES
# ==========================================

chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.json"))

if not chunk_files:

    print("No chunk files found!")
    print(f"Check this folder: {CHUNKS_DIR}")
    exit()


print(f"\nFound {len(chunk_files)} chunk files.")


# ==========================================
# LOAD ALL CHUNKS
# ==========================================

all_chunks = []

for file in chunk_files:

    print(f"Loading: {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    all_chunks.extend(chunks)


print("\n================================")
print("ALL CHUNKS LOADED")
print("================================")
print(f"Chunk files : {len(chunk_files)}")
print(f"Total chunks: {len(all_chunks)}")


# ==========================================
# CHECK CHUNKS
# ==========================================

if not all_chunks:

    print("No chunks available!")
    exit()


# ==========================================
# EXTRACT TEXT
# ==========================================

texts = [
    chunk["text"]
    for chunk in all_chunks
]


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

print("\n================================")
print("CREATING EMBEDDINGS")
print("================================")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)


# ==========================================
# NORMALIZE EMBEDDINGS
# ==========================================

faiss.normalize_L2(embeddings)


# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

print(f"\nEmbedding dimension: {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# ==========================================
# CREATE VECTORSTORE FOLDER
# ==========================================

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# SAVE FAISS INDEX
# ==========================================

index_path = VECTORSTORE_DIR / "index.faiss"

faiss.write_index(
    index,
    str(index_path)
)


# ==========================================
# SAVE METADATA
# ==========================================

metadata_path = VECTORSTORE_DIR / "metadata.pkl"

with open(metadata_path, "wb") as f:

    pickle.dump(
        all_chunks,
        f
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("VECTOR DATABASE CREATED")
print("================================")

print(f"FAISS index : {index_path}")
print(f"Metadata    : {metadata_path}")
print(f"Vectors     : {index.ntotal}")
print(f"Dimensions  : {dimension}")

print("\nAll lectures are now searchable!")


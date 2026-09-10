import json
import os
from pathlib import Path


# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = Path(r"D:\AI_Projects(RAG)")

INPUT_DIR = BASE_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "chunks"


# -----------------------------
# SETTINGS
# -----------------------------

MAX_WORDS = 180
OVERLAP_WORDS = 30


# -----------------------------
# LOAD TRANSCRIPT
# -----------------------------

def load_transcript(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# CREATE CHUNKS
# -----------------------------

def create_chunks(segments, source_name):

    chunks = []

    current_text = []
    current_start = None
    current_end = None

    chunk_id = 1

    for segment in segments:

        text = segment["text"].strip()

        if not text:
            continue

        words = text.split()

        # Start new chunk
        if current_start is None:
            current_start = segment["start"]

        current_text.extend(words)
        current_end = segment["end"]

        # Check chunk size
        if len(current_text) >= MAX_WORDS:

            chunk_text = " ".join(current_text)

            chunks.append({
                "chunk_id": chunk_id,
                "source": source_name,
                "text": chunk_text,
                "start": current_start,
                "end": current_end
            })

            chunk_id += 1

            # Keep overlap
            current_text = current_text[-OVERLAP_WORDS:]

            current_start = segment["end"]

    # Save remaining text
    if current_text:

        chunks.append({
            "chunk_id": chunk_id,
            "source": source_name,
            "text": " ".join(current_text),
            "start": current_start,
            "end": current_end
        })

    return chunks


# -----------------------------
# PROCESS ALL JSON FILES
# -----------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_DIR.glob("*.json"))

    if not json_files:

        print("No JSON transcript files found.")

        return

    for file_path in json_files:

        print(f"\nProcessing: {file_path.name}")

        segments = load_transcript(file_path)

        chunks = create_chunks(
            segments,
            file_path.stem
        )

        output_file = OUTPUT_DIR / f"{file_path.stem}_chunks.json"

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(f"Created {len(chunks)} chunks")
        print(f"Saved: {output_file}")

    print("\n==============================")
    print("CHUNKING COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()
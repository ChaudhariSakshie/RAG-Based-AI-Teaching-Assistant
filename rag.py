
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(r"D:\AI_Projects(RAG)")
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"


# ==========================================
# SETTINGS
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

TOP_K = 5

# Minimum similarity required for a result
SIMILARITY_THRESHOLD = 0.35


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ==========================================
# LOAD FAISS DATABASE
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print(
    f"FAISS index loaded. "
    f"Vectors: {index.ntotal}"
)


# ==========================================
# LOAD METADATA
# ==========================================

with open(
    METADATA_PATH,
    "rb"
) as f:

    metadata = pickle.load(f)


print(
    f"Metadata loaded. "
    f"Chunks: {len(metadata)}"
)


# ==========================================
# RETRIEVER
# ==========================================

def retrieve(query, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Normalize query embedding
    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        score = float(score)

        # Ignore weak matches
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = metadata[idx]

        results.append({

            "score": score,

            "source": chunk["source"],

            "chunk_id": chunk["chunk_id"],

            "start": chunk["start"],

            "end": chunk["end"],

            "text": chunk["text"]
        })

    return results


# ==========================================
# BUILD CONTEXT
# ==========================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(

            f"""
SOURCE {i}

Lecture:
{result['source']}

Timestamp:
{result['start']:.2f} - {result['end']:.2f} seconds

Content:
{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ==========================================
# ASK MISTRAL
# ==========================================

def ask_llm(
    question,
    context
):

    prompt = f"""
You are a strict lecture-based AI Teaching Assistant.

Your job is to answer the student's question
ONLY from the lecture content provided below.

IMPORTANT RULES:

1. Use ONLY the information present in the
   provided lecture content.

2. DO NOT use your own general knowledge.

3. DO NOT add examples, facts, explanations,
   calculations, definitions, or steps that are
   not explicitly supported by the lecture content.

4. DO NOT guess or assume missing information.

5. If the lecture content does not contain
   enough information to answer the question,
   respond exactly with:

"I could not find this information
in the provided lecture."

6. If only part of the answer is available,
   provide ONLY that part and clearly state
   what information is missing.

7. Keep the answer simple and
   student-friendly.

8. When possible, mention the lecture number
   and timestamp where the answer was found.

LECTURE CONTENT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = chat(

        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==========================================
# MAIN RAG SYSTEM
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("AI TEACHING ASSISTANT")
    print("================================")

    question = input(
        "\nAsk a question about the lectures: "
    )

    if not question.strip():

        print(
            "\nPlease enter a question."
        )

        exit()


    # --------------------------------------
    # RETRIEVE RELEVANT CHUNKS
    # --------------------------------------

    print(
        "\nSearching lecture content..."
    )

    results = retrieve(
        question,
        top_k=TOP_K
    )


    # --------------------------------------
    # NO RELEVANT RESULTS
    # --------------------------------------

    if not results:

        print("\n================================")
        print("ANSWER")
        print("================================")

        print(
            "I could not find this information "
            "in the provided lecture."
        )

        print("\nNo sufficiently relevant sources found.")

        exit()


    # --------------------------------------
    # DISPLAY RETRIEVAL INFO
    # --------------------------------------

    print(
        f"\nFound {len(results)} relevant sources."
    )


    # --------------------------------------
    # BUILD CONTEXT
    # --------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------
    # GENERATE ANSWER
    # --------------------------------------

    print(
        "\nGenerating answer using Mistral..."
    )

    answer = ask_llm(
        question,
        context
    )


    # ======================================
    # DISPLAY FINAL ANSWER
    # ======================================

    print("\n================================")
    print("ANSWER")
    print("================================")

    print(answer)


    # ======================================
    # DISPLAY SOURCES
    # ======================================

    print("\n================================")
    print("SOURCES")
    print("================================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nSource {i}: "
            f"{result['source']}"
        )

        print(
            f"Timestamp: "
            f"{result['start']:.2f}s - "
            f"{result['end']:.2f}s"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )


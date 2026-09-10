# AI Teaching Assistant — RAG Based Learning System

An AI-powered Teaching Assistant that allows students to ask questions about lecture content and receive answers grounded in the provided educational lectures.

## Features

- YouTube lecture processing
- Speech-to-text transcription using Whisper
- Transcript cleaning and chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector database for similarity search
- Local Mistral LLM using Ollama
- Retrieval-Augmented Generation (RAG)
- Lecture-specific question answering
- Source lecture and timestamp display
- Semantic similarity scores
- Lecture summaries
- Key points
- Study notes
- Quiz generation and scoring
- Chat history
- Downloadable summaries and notes
- Streamlit web interface
- Local AI processing

## Technology Stack

- Python
- Streamlit
- OpenAI Whisper
- Sentence Transformers
- FAISS
- Ollama
- Mistral
- Pandas
- NumPy
- Matplotlib
- Seaborn

## RAG Pipeline

```text
YouTube Lectures
       ↓
Audio Extraction
       ↓
Whisper Transcription
       ↓
Transcript Cleaning
       ↓
Text Chunking
       ↓
Sentence Transformers
       ↓
Vector Embeddings
       ↓
FAISS Vector Database
       ↓
Relevant Lecture Retrieval
       ↓
Mistral LLM
       ↓
Grounded Answer
       ↓
Streamlit AI Teaching Assistant


Project structure
AI_Projects(RAG)
│
├── audios/
├── transcripts/
├── cleaned/
├── chunks/
├── vectorstore/
│   ├── index.faiss
│   └── metadata.pkl
├── src/
│   ├── cleaning.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── rag.py
├── tests/
├── Videos/
├── whisper/
├── stt.py
├── app.py
├── requirements.txt
└── README.md
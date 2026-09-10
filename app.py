import pickle
import json
import re
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import ollama


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(r"D:\AI_Projects(RAG)")
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

TOP_K = 5
SEARCH_K = 20
SIMILARITY_THRESHOLD = 0.35


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #aab2c0;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .stat-card {
        background-color: #171b24;
        border: 1px solid #2b3240;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        min-height: 105px;
    }

    .stat-number {
        font-size: 28px;
        font-weight: 700;
    }

    .stat-label {
        color: #aab2c0;
        font-size: 14px;
    }

    .answer-box {
        background-color: #151922;
        border: 1px solid #303747;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .feature-card {
        background-color: #151922;
        border: 1px solid #303747;
        border-radius: 12px;
        padding: 18px;
        min-height: 120px;
    }

    .quiz-card {
        background-color: #151922;
        border: 1px solid #303747;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 AI Teaching Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered learning assistant for lecture-based '
    'question answering, summaries, notes and quizzes using '
    '<b>Retrieval-Augmented Generation (RAG)</b>.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    index = faiss.read_index(
        str(VECTORSTORE_DIR / "index.faiss")
    )

    with open(
        VECTORSTORE_DIR / "metadata.pkl",
        "rb"
    ) as f:
        metadata = pickle.load(f)

    return embedding_model, index, metadata


with st.spinner("Loading AI Teaching Assistant..."):

    try:

        embedding_model, index, metadata = load_resources()

    except Exception as e:

        st.error("Unable to load the RAG system.")
        st.exception(e)
        st.stop()


# =========================================================
# LECTURE INFORMATION
# =========================================================

lecture_names = sorted(
    list(
        set(
            item["source"]
            for item in metadata
        )
    )
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "notes" not in st.session_state:
    st.session_state.notes = ""

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📚 Knowledge Base")

    st.metric(
        "🎥 Lectures",
        len(lecture_names)
    )

    st.metric(
        "🧩 Knowledge Chunks",
        len(metadata)
    )

    st.metric(
        "🔢 Vector Dimensions",
        index.d
    )

    st.divider()

    st.markdown("## 🤖 AI Pipeline")

    st.write("🎥 Lecture Videos")
    st.write("↓")
    st.write("📝 Whisper")
    st.write("↓")
    st.write("✂️ Chunking")
    st.write("↓")
    st.write("🧠 Embeddings")
    st.write("↓")
    st.write("🔎 FAISS")
    st.write("↓")
    st.write("🤖 Mistral")

    st.divider()

    st.markdown("## 🛡️ Grounding")

    st.caption(
        "Answers, summaries, notes and quizzes "
        "are generated from the indexed lecture content."
    )

    st.divider()

    if st.button(
        "🧹 Clear Session",
        use_container_width=True
    ):

        st.session_state.chat_history = []
        st.session_state.last_results = []
        st.session_state.summary = ""
        st.session_state.notes = ""
        st.session_state.quiz_data = []
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0

        st.rerun()


# =========================================================
# DASHBOARD
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{len(lecture_names)}</div>
            <div class="stat-label">Lectures Indexed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{len(metadata)}</div>
            <div class="stat-label">Knowledge Chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{index.d}</div>
            <div class="stat-label">Vector Dimensions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">Mistral</div>
            <div class="stat-label">Local AI Model</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# LECTURE SELECTION
# =========================================================

st.markdown("### 📖 Select Lecture")

lecture_options = ["All Lectures"] + lecture_names

selected_lecture = st.selectbox(
    "Choose the knowledge source",
    lecture_options,
    key="selected_lecture"
)


# =========================================================
# GET LECTURE TEXT
# =========================================================

def get_lecture_text(lecture_name):

    lecture_chunks = [
        item
        for item in metadata
        if item["source"] == lecture_name
    ]

    lecture_chunks = sorted(
        lecture_chunks,
        key=lambda x: x["start"]
    )

    return "\n".join(
        item["text"]
        for item in lecture_chunks
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(question):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Search the complete FAISS index first.
    # Then apply the selected lecture filter so relevant chunks
    # from a selected lecture are not lost from the global top-K.
    search_k = index.ntotal

    scores, indices = index.search(
        question_embedding,
        search_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        similarity = float(score)

        if similarity < SIMILARITY_THRESHOLD:
            continue

        item = metadata[idx]

        if (
            selected_lecture != "All Lectures"
            and item["source"] != selected_lecture
        ):
            continue

        results.append(
            {
                "score": similarity,
                "source": item["source"],
                "text": item["text"],
                "start": item["start"],
                "end": item["end"]
            }
        )

        if len(results) >= TOP_K:
            break

    return results



# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, results):

    fallback = "I could not find this information in the provided lectures."

    if not results:
        return fallback

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

Lecture:
{result["source"]}

Timestamp:
{result["start"]:.2f}s - {result["end"]:.2f}s

Lecture Content:
{result["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an AI Teaching Assistant for a lecture-based RAG system.

Your task is to answer the student's question ONLY from the
LECTURE CONTENT supplied below.

This is a strict source-grounded task.

GROUNDING RULES:

1. Use ONLY facts, definitions, explanations, examples and
   calculations explicitly present in the supplied lecture content.

2. Do NOT use your pretrained knowledge or outside knowledge.

3. Do NOT solve, calculate, complete, or infer an answer that
   the lecture does not explicitly provide.

4. If the lecture gives a method but does not explicitly give
   the final result, do NOT calculate the final result yourself.

5. Do NOT add common examples that are not present in the lecture.

6. Do NOT add related concepts, methods, functions, libraries,
   applications, advantages or disadvantages unless they are
   explicitly supported by the lecture content.

7. Do NOT expand a short lecture statement with information
   you already know.

8. If the transcript wording is unclear, preserve what is
   supported and do not guess or silently correct it.

9. If the supplied sources only partially answer the question,
   give ONLY the supported portion.

10. If the supplied lecture content does not contain enough
    information to answer the question, respond EXACTLY:

I could not find this information in the provided lectures.

11. Keep supported answers concise and student-friendly.

IMPORTANT:
The similarity score only indicates semantic similarity between
the question and a retrieved lecture chunk. It does NOT prove
that every fact in the chunk answers the question.

LECTURE CONTENT:

{context}

STUDENT QUESTION:

{question}

GROUNDED ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"].strip()

    # Prevent obvious model refusal variants from becoming
    # inconsistent fallback messages.
    if not answer:
        return fallback

    return answer



# =========================================================
# GENERATE SUMMARY
# =========================================================

def generate_summary(lecture_name):

    lecture_text = get_lecture_text(
        lecture_name
    )

    if not lecture_text.strip():

        return (
            "No content found for this lecture."
        )

    prompt = f"""
You are an AI Teaching Assistant.

Create a study summary using ONLY
the lecture transcript below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not add facts not present in the transcript.
3. Do not invent examples.
4. Do not introduce concepts not discussed.
5. Preserve the meaning of the lecture.
6. If transcription wording is unclear,
   do not invent a correction.
7. Keep the summary concise and student-friendly.

Use this format:

## Summary

Write a concise paragraph based only
on the lecture.

## Key Points

- Point supported by the lecture
- Point supported by the lecture
- Point supported by the lecture
- Point supported by the lecture
- Point supported by the lecture

LECTURE TRANSCRIPT:

{lecture_text}

SUMMARY:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ].strip()


# =========================================================
# GENERATE STUDY NOTES
# =========================================================

def generate_notes(lecture_name):

    lecture_text = get_lecture_text(
        lecture_name
    )

    if not lecture_text.strip():

        return "No content found."

    prompt = f"""
You are an AI Teaching Assistant.

Create short exam-oriented study notes
from ONLY the lecture transcript.

STRICT RULES:

1. Use only information in the transcript.
2. Do not use outside knowledge.
3. Do not invent definitions.
4. Do not add concepts not discussed.
5. Preserve the terminology used in the lecture.
6. Keep the notes easy to revise.

Use this format:

# Study Notes

## Important Concepts

- ...
- ...
- ...

## Important Explanations

- ...
- ...
- ...

## Examples Mentioned in Lecture

- ...
- ...

## Quick Revision

Write 3-5 short revision statements.

LECTURE:

{lecture_text}

NOTES:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ].strip()


# =========================================================
# GENERATE QUIZ
# =========================================================

def generate_quiz(lecture_name, number_of_questions):

    lecture_text = get_lecture_text(
        lecture_name
    )

    if not lecture_text.strip():

        return []

    prompt = f"""
You are an AI Teaching Assistant.

Create {number_of_questions} multiple-choice
questions using ONLY the lecture transcript below.

STRICT RULES:

1. Every question must be answerable from
   the lecture transcript.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Do not create questions about concepts
   not discussed in the lecture.

5. Each question must have exactly four options.

6. Only one option must be correct.

7. Include a short explanation based only
   on the lecture.

8. Return ONLY valid JSON.

Use exactly this JSON structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": 0,
    "explanation": "Explanation supported by lecture"
  }}
]

The answer value must be:
0 for first option,
1 for second option,
2 for third option,
3 for fourth option.

LECTURE:

{lecture_text}

JSON:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response[
        "message"
    ][
        "content"
    ].strip()

    try:

        # Remove markdown JSON fences if Mistral adds them
        raw = re.sub(
            r"```json\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"```\s*$",
            "",
            raw
        )

        start = raw.find("[")

        end = raw.rfind("]")

        if start == -1 or end == -1:

            return []

        raw = raw[
            start:end + 1
        ]

        quiz = json.loads(raw)

        valid_questions = []

        for item in quiz:

            if not isinstance(item, dict):
                continue

            question = item.get(
                "question",
                ""
            )

            options = item.get(
                "options",
                []
            )

            answer = item.get(
                "answer",
                -1
            )

            explanation = item.get(
                "explanation",
                ""
            )

            if (
                question
                and isinstance(options, list)
                and len(options) == 4
                and isinstance(answer, int)
                and 0 <= answer <= 3
            ):

                valid_questions.append(
                    {
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "explanation": explanation
                    }
                )

        return valid_questions

    except Exception:

        return []


# =========================================================
# Q&A SECTION
# =========================================================

st.divider()

st.markdown("## 💬 Ask Your Lecture")

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is a lambda function in Python?"
    ),
    key="question_input"
)


if st.button(
    "🔍 Ask AI",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🔎 Searching lecture knowledge base..."
        ):

            results = retrieve_context(
                question
            )

        st.session_state.last_results = results

        if not results:

            answer = (
                "I could not find this information "
                "in the provided lectures."
            )

        else:

            with st.spinner(
                "🤖 Mistral is generating a grounded answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        results
                    )

                except Exception as e:

                    st.error(
                        "Mistral could not generate the answer."
                    )

                    st.exception(e)

                    answer = ""


        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )


# =========================================================
# CHAT HISTORY
# =========================================================

if st.session_state.chat_history:

    st.markdown("## 🤖 AI Answers")

    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"**You:** {chat['question']}"
        )

        st.markdown(
            f"""
            <div class="answer-box">
            {chat['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# SOURCES
# =========================================================

if st.session_state.last_results:

    st.markdown(
        "## 📚 Retrieved Lecture Sources"
    )

    st.caption(
        f"Found {len(st.session_state.last_results)} "
        "relevant lecture sections."
    )

    for i, result in enumerate(
        st.session_state.last_results,
        start=1
    ):

        similarity_percent = (
            result["score"] * 100
        )

        title = (
            f"Source {i} • "
            f"{result['source']} • "
            f"{similarity_percent:.1f}% semantic similarity"
        )

        with st.expander(title):

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "⏱️ **Timestamp**"
                )

                st.write(
                    f"{result['start']:.2f}s → "
                    f"{result['end']:.2f}s"
                )

            with col2:

                st.markdown(
                    "📊 **Semantic Similarity**"
                )

                st.write(
                    f"{similarity_percent:.1f}%"
                )

            st.divider()

            st.markdown(
                "📝 **Lecture Content**"
            )

            st.write(
                result["text"]
            )


# =========================================================
# LEARNING TOOLS
# =========================================================

st.divider()

st.markdown(
    "## 📖 Lecture Learning Tools"
)

st.caption(
    "Turn your lecture into summaries, notes and quizzes."
)


tool_col1, tool_col2, tool_col3 = st.columns(3)


# =========================================================
# SUMMARY
# =========================================================

with tool_col1:

    st.markdown(
        """
        <div class="feature-card">
        <h4>📝 Summary</h4>
        Generate a concise lecture summary
        and important key points.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        if selected_lecture == "All Lectures":

            st.warning(
                "Select a specific lecture first."
            )

        else:

            with st.spinner(
                "🤖 Generating lecture summary..."
            ):

                try:

                    st.session_state.summary = (
                        generate_summary(
                            selected_lecture
                        )
                    )

                except Exception as e:

                    st.error(
                        "Unable to generate summary."
                    )

                    st.exception(e)


# =========================================================
# NOTES
# =========================================================

with tool_col2:

    st.markdown(
        """
        <div class="feature-card">
        <h4>📚 Study Notes</h4>
        Generate exam-oriented notes
        from the selected lecture.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Notes",
        use_container_width=True
    ):

        if selected_lecture == "All Lectures":

            st.warning(
                "Select a specific lecture first."
            )

        else:

            with st.spinner(
                "🤖 Creating study notes..."
            ):

                try:

                    st.session_state.notes = (
                        generate_notes(
                            selected_lecture
                        )
                    )

                except Exception as e:

                    st.error(
                        "Unable to generate notes."
                    )

                    st.exception(e)


# =========================================================
# QUIZ
# =========================================================

with tool_col3:

    st.markdown(
        """
        <div class="feature-card">
        <h4>🎯 Quiz</h4>
        Generate MCQs from the selected
        lecture and test your knowledge.
        </div>
        """,
        unsafe_allow_html=True
    )

    number_of_questions = st.selectbox(
        "Questions",
        [5, 10, 15],
        key="quiz_question_count"
    )

    if st.button(
        "Generate Quiz",
        use_container_width=True
    ):

        if selected_lecture == "All Lectures":

            st.warning(
                "Select a specific lecture first."
            )

        else:

            with st.spinner(
                "🤖 Generating quiz..."
            ):

                try:

                    quiz = generate_quiz(
                        selected_lecture,
                        number_of_questions
                    )

                    if quiz:

                        st.session_state.quiz_data = quiz
                        st.session_state.quiz_submitted = False

                    else:

                        st.error(
                            "Could not generate a valid quiz. "
                            "Please try again."
                        )

                except Exception as e:

                    st.error(
                        "Unable to generate quiz."
                    )

                    st.exception(e)


# =========================================================
# DISPLAY SUMMARY
# =========================================================

if st.session_state.summary:

    st.divider()

    st.markdown(
        f"## 📝 Summary — {selected_lecture}"
    )

    st.markdown(
        st.session_state.summary
    )

    st.download_button(
        "⬇️ Download Summary",
        data=st.session_state.summary,
        file_name=f"{selected_lecture}_summary.txt",
        mime="text/plain"
    )


# =========================================================
# DISPLAY NOTES
# =========================================================

if st.session_state.notes:

    st.divider()

    st.markdown(
        f"## 📚 Study Notes — {selected_lecture}"
    )

    st.markdown(
        st.session_state.notes
    )

    st.download_button(
        "⬇️ Download Notes",
        data=st.session_state.notes,
        file_name=f"{selected_lecture}_study_notes.txt",
        mime="text/plain"
    )


# =========================================================
# QUIZ DISPLAY
# =========================================================

if st.session_state.quiz_data:

    st.divider()

    st.markdown(
        f"## 🎯 Quiz — {selected_lecture}"
    )

    st.caption(
        f"{len(st.session_state.quiz_data)} questions"
    )

    quiz_answers = {}

    for i, item in enumerate(
        st.session_state.quiz_data
    ):

        st.markdown(
            f"""
            <div class="quiz-card">
            <b>Question {i + 1}</b><br><br>
            {item["question"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        quiz_answers[i] = st.radio(
            "Choose an answer:",
            item["options"],
            key=f"quiz_{i}",
            index=None
        )

    if st.button(
        "✅ Submit Quiz",
        type="primary"
    ):

        score = 0

        for i, item in enumerate(
            st.session_state.quiz_data
        ):

            selected_answer = quiz_answers.get(
                i
            )

            correct_answer = item[
                "options"
            ][
                item["answer"]
            ]

            if selected_answer == correct_answer:

                score += 1

        st.session_state.quiz_submitted = True
        st.session_state.quiz_score = score


# =========================================================
# QUIZ RESULTS
# =========================================================

if (
    st.session_state.quiz_submitted
    and st.session_state.quiz_data
):

    score = st.session_state.quiz_score
    total = len(
        st.session_state.quiz_data
    )

    percentage = (
        score / total
    ) * 100

    st.divider()

    st.markdown("## 🏆 Quiz Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Score",
            f"{score}/{total}"
        )

    with result_col2:

        st.metric(
            "Percentage",
            f"{percentage:.1f}%"
        )

    if percentage >= 80:

        st.success(
            "Excellent! You have a strong understanding "
            "of this lecture."
        )

    elif percentage >= 60:

        st.info(
            "Good job! Review the incorrect answers "
            "to strengthen your understanding."
        )

    else:

        st.warning(
            "Review the lecture notes and try the quiz again."
        )

    st.markdown(
        "### 📖 Answer Review"
    )

    for i, item in enumerate(
        st.session_state.quiz_data
    ):

        correct = item[
            "options"
        ][
            item["answer"]
        ]

        st.markdown(
            f"""
            **Q{i + 1}. {item["question"]}**

            ✅ Correct Answer: **{correct}**

            💡 Explanation: {item["explanation"]}
            """
        )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.markdown(
    "## 🚀 About This AI Teaching Assistant"
)

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.markdown(
        """
        ### 🔧 Technology Stack

        - **Whisper** — Lecture transcription
        - **Python** — Application logic
        - **Sentence Transformers** — Text embeddings
        - **FAISS** — Vector similarity search
        - **Mistral** — Local LLM
        - **Streamlit** — Web application
        - **RAG** — Grounded question answering
        """
    )

with about_col2:

    st.markdown(
        """
        ### ✨ Features

        - 💬 Lecture-based Q&A
        - 🔎 Semantic search
        - 📚 Source retrieval
        - ⏱️ Timestamp information
        - 📝 Lecture summaries
        - 📖 Study notes
        - 🎯 AI-generated quizzes
        - 🏆 Quiz scoring
        - ⬇️ Downloadable study material
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 AI Teaching Assistant | "
    "Whisper + Sentence Transformers + FAISS + Mistral | "
    "Retrieval-Augmented Generation"
)
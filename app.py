
import io
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone

import streamlit as st


APP_NAME = "Omnis AI"
MIN_SCORE = 0.10
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35

st.set_page_config(
    page_title="Omnis AI",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clean_text(text):
    """Normalize whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Extract simple lowercase word tokens."""
    return re.findall(r"[a-zA-Z0-9_'-]+", text.lower())


def split_into_chunks(text, filename):
    """Split a document into overlapping, searchable sections."""
    words = clean_text(text).split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        content = " ".join(words[start:end])

        if content.strip():
            chunks.append({
                "source": filename,
                "text": content,
                "start_word": start,
            })

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def read_uploaded_file(uploaded_file):
    """Read supported text files safely."""
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        return clean_text(content)
    except UnicodeDecodeError:
        return ""


def build_knowledge_base(uploaded_files):
    """Create chunks from uploaded files."""
    documents = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        extension = name.lower().rsplit(".", 1)[-1]

        if extension not in {"txt", "md"}:
            continue

        text = read_uploaded_file(uploaded_file)

        if text:
            documents.extend(split_into_chunks(text, name))

    return documents


def vectorize(text):
    """Build a normalized term-frequency vector."""
    words = tokenize(text)
    counts = Counter(words)
    total = sum(counts.values()) or 1

    return {word: count / total for word, count in counts.items()}


def build_search_index(chunks):
    """Build a lightweight TF-IDF index using Python only."""
    token_lists = [tokenize(chunk["text"]) for chunk in chunks]
    document_count = len(chunks)
    document_frequency = Counter()

    for tokens in token_lists:
        document_frequency.update(set(tokens))

    idf = {
        word: math.log(1 + document_count / (1 + count)) + 1
        for word, count in document_frequency.items()
    }

    vectors = []

    for tokens in token_lists:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector = {
            word: (count / total) * idf.get(word, 1.0)
            for word, count in counts.items()
        }

        length = math.sqrt(sum(value * value for value in vector.values()))

        if length:
            vector = {
                word: value / length
                for word, value in vector.items()
            }

        vectors.append(vector)

    return idf, vectors


def search_documents(question, chunks, idf, vectors):
    """Rank document chunks by cosine similarity."""
    query_counts = Counter(tokenize(question))
    query_total = sum(query_counts.values()) or 1

    query_vector = {
        word: (count / query_total) * idf.get(word, 1.0)
        for word, count in query_counts.items()
        if word in idf
    }

    query_length = math.sqrt(
        sum(value * value for value in query_vector.values())
    )

    if not query_length:
        return []

    query_vector = {
        word: value / query_length
        for word, value in query_vector.items()
    }

    ranked = []

    for index, document_vector in enumerate(vectors):
        score = sum(
            value * document_vector.get(word, 0.0)
            for word, value in query_vector.items()
        )

        if score > 0:
            ranked.append((score, chunks[index]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:3]


def log_unresolved(question, best_score):
    """Append unanswered questions to a local JSONL file."""
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "best_score": round(best_score, 4),
    }

    try:
        with open("unresolved_questions.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def make_answer(question, ranked):
    """Return a cautious answer based on retrieved evidence."""
    if not ranked:
        log_unresolved(question, 0.0)
        return (
            "I couldn't find matching information in the current documents. "
            "Try different wording or upload a document containing the answer."
        ), []

    best_score = ranked[0][0]

    if best_score < MIN_SCORE:
        log_unresolved(question, best_score)
        return (
            "I couldn't find strong enough evidence to answer reliably. "
            "Please add a relevant document or ask a more specific question."
        ), ranked

    # This version retrieves evidence; it does not generate new facts.
    answer = (
        "I found relevant information in your knowledge base. "
        "Review the evidence below to find the answer to your question."
    )

    return answer, ranked


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b1120;
            color: #edf2f7;
        }
        [data-testid="stSidebar"] {
            background: #101a2c;
            border-right: 1px solid #25344b;
        }
        .omnis-brand {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -1px;
            color: #72e0c1;
            margin-bottom: 0;
        }
        .omnis-subtitle {
            color: #a8b8ca;
            margin-top: 4px;
        }
        .evidence-card {
            background: #111d30;
            border: 1px solid #2c4059;
            border-radius: 12px;
            padding: 16px;
            margin: 10px 0;
        }
        .source-name {
            color: #72e0c1;
            font-weight: 700;
        }
        .small-note {
            color: #a8b8ca;
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] {
            background: #111d30;
            border: 1px solid #25344b;
            border-radius: 12px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "evaluation" not in st.session_state:
    st.session_state.evaluation = []


with st.sidebar:
    st.markdown("## ◎ Omnis AI")
    st.caption("Evidence-first document assistant")
    st.divider()

    st.markdown("### Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload TXT or Markdown files",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="Files are processed as searchable text.",
    )

    if st.button("Load uploaded documents", use_container_width=True):
        st.session_state.documents = build_knowledge_base(uploaded_files or [])
        st.session_state.messages = []

        if st.session_state.documents:
            st.success(
                f"Loaded {len(st.session_state.documents)} text sections."
            )
        else:
            st.warning("No readable TXT or Markdown content was found.")

    if st.button("Load built-in sample", use_container_width=True):
        sample = """
        Omnis AI is a document question-answering assistant.
        It divides documents into smaller chunks and ranks them against questions.
        Keyword frequency and inverse document frequency help find relevant chunks.
        Supported answers should show a source filename and a passage as evidence.
        The system should not invent facts missing from its knowledge base.
        Weak evidence should produce an uncertainty message.
        Retrieval quality can be evaluated using questions with known answers.
        Unanswered questions can be logged for future improvement.
        """
        st.session_state.documents = split_into_chunks(
            sample, "Built-in sample"
        )
        st.session_state.messages = []
        st.success("Sample knowledge base loaded.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    st.divider()
    st.caption("Supported files: .txt and .md")
    st.caption("Retrieval runs without a paid AI API.")


st.markdown('<p class="omnis-brand">◎ Omnis AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="omnis-subtitle">Answers grounded in your documents.</p>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Text sections", len(st.session_state.documents))

with metric2:
    st.metric(
        "Questions asked",
        sum(1 for message in st.session_state.messages
            if message["role"] == "user"),
    )

with metric3:
    st.metric("Engine", "Local TF-IDF")


st.info(
    "Upload documents from the sidebar or load the built-in sample. "
    "Ask a question and inspect the cited evidence."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("evidence"):
            for number, item in enumerate(message["evidence"], start=1):
                score = item["score"]
                chunk = item["chunk"]

                st.markdown(
                    f"""
                    <div class="evidence-card">
                      <div class="source-name">
                        Source {number}: {chunk['source']}
                      </div>
                      <div class="small-note">
                        Retrieval similarity: {score:.3f}
                      </div>
                      <p>{chunk['text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    if not st.session_state.documents:
        answer = "Please load the sample or upload documents first."
        evidence = []
    else:
        idf, vectors = build_search_index(st.session_state.documents)
        ranked = search_documents(
            question,
            st.session_state.documents,
            idf,
            vectors,
        )
        answer, results = make_answer(question, ranked)

        evidence = [
            {"score": score, "chunk": chunk}
            for score, chunk in results
        ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": evidence,
    })

    st.rerun()


st.divider()
st.markdown("### Retrieval evaluation")

st.write(
    "Test whether Omnis AI retrieves the expected source for a known question."
)

with st.form("evaluation_form"):
    eval_question = st.text_input(
        "Test question",
        placeholder="What does the system do when evidence is weak?",
    )
    expected_source = st.text_input(
        "Expected source filename",
        placeholder="Built-in sample",
    )
    submitted = st.form_submit_button("Run evaluation")

if submitted:
    if not eval_question.strip() or not expected_source.strip():
        st.warning("Enter both a question and the expected source.")
    elif not st.session_state.documents:
        st.warning("Load a knowledge base before evaluating.")
    else:
        idf, vectors = build_search_index(st.session_state.documents)
        results = search_documents(
            eval_question,
            st.session_state.documents,
            idf,
            vectors,
        )

        retrieved_source = results[0][1]["source"] if results else "None"
        passed = retrieved_source.lower() == expected_source.strip().lower()

        result = {
            "question": eval_question,
            "expected_source": expected_source.strip(),
            "retrieved_source": retrieved_source,
            "passed": passed,
        }

        st.session_state.evaluation.append(result)

        if passed:
            st.success(f"PASS — retrieved {retrieved_source}")
        else:
            st.warning(
                f"Not a match — retrieved {retrieved_source}. "
                "Check the expected filename or improve the knowledge base."
            )

if st.session_state.evaluation:
    st.markdown("#### Evaluation history")
    st.dataframe(st.session_state.evaluation, use_container_width=True)

    evaluation_json = json.dumps(
        st.session_state.evaluation,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "Download evaluation results",
        data=evaluation_json,
        file_name="omnis_ai_evaluation.json",
        mime="application/json",
    )

st.divider()
st.caption(
    "Omnis AI • Evidence-first retrieval • Built for clarity"
      )

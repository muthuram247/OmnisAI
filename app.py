import json
import math
import re
from collections import Counter
from datetime import datetime, timezone

import streamlit as st
from google import genai

APP_NAME = "Omnis AI"
MODEL_NAME = "gemini-2.5-flash"
MIN_SCORE = 0.10
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35
MAX_HISTORY_MESSAGES = 10

st.set_page_config(
    page_title=APP_NAME,
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- FOREST GREEN & CREAM THEME --------------------

st.markdown(
    """
    <style>
    :root {
        --forest: #173F35;
        --forest-hover: #245748;
        --cream: #F7F3E8;
        --surface: #FFFCF5;
        --sage: #E4EBDD;
        --border: #DCDCCF;
        --text: #24342D;
        --muted: #737C70;
    }

    .stApp {
        background: #F7F3E8 !important;
        color: #24342D !important;
    }

    .main .block-container {
        max-width: 1050px;
        padding: 2.2rem 2rem 3rem;
    }

    [data-testid="stSidebar"] {
        background: #EDEFE4 !important;
        border-right: 1px solid #DCDCCF;
    }

    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #173F35 !important;
    }

    .brand {
        color: #173F35 !important;
        font-size: 34px;
        font-weight: 750;
        letter-spacing: -1.3px;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #737C70 !important;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    h1, h2, h3, h4 {
        color: #173F35 !important;
        letter-spacing: -0.35px;
    }

    p, li, label {
        color: #24342D;
    }

    [data-testid="stCaptionContainer"] {
        color: #737C70 !important;
    }

    /* CHAT MESSAGES */

    [data-testid="stChatMessage"] {
        background: #FFFCF5 !important;
        border: 1px solid #DCDCCF;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: 0.85rem 0;
        box-shadow: 0 2px 8px rgba(23, 63, 53, 0.035);
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.75;
    }

    /* CHAT INPUT: FORCE LIGHT BACKGROUND */

    [data-testid="stChatInput"] {
        background: #FFFCF5 !important;
        border: 2px solid #B9C8B6 !important;
        border-radius: 20px !important;
        box-shadow: 0 5px 18px rgba(23, 63, 53, 0.08) !important;
        overflow: hidden !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #173F35 !important;
        box-shadow: 0 0 0 3px rgba(23, 63, 53, 0.12) !important;
    }

    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: #FFFCF5 !important;
        color: #24342D !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        background-color: #FFFCF5 !important;
        color: #24342D !important;
        -webkit-text-fill-color: #24342D !important;
        caret-color: #173F35 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 13px 12px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #737C70 !important;
        -webkit-text-fill-color: #737C70 !important;
        opacity: 1 !important;
    }

    /* SEND BUTTON: GREEN WITH WHITE ARROW */

    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {
        background: #173F35 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInputSubmitButton"]:hover {
        background: #245748 !important;
    }

    [data-testid="stChatInput"] button:disabled,
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: #527568 !important;
        color: #FFFFFF !important;
        opacity: 0.85 !important;
    }

    /* BUTTONS */

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        background: #FFFCF5 !important;
        color: #173F35 !important;
        border: 1px solid #B9C8B6 !important;
        border-radius: 12px !important;
        min-height: 43px;
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #E4EBDD !important;
        color: #173F35 !important;
        border-color: #173F35 !important;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: #173F35 !important;
        color: #FFFFFF !important;
        border-color: #173F35 !important;
    }

    /* TEXT INPUTS */

    .stTextInput input,
    .stTextArea textarea,
    [data-testid="stTextInput"] input {
        background: #FFFCF5 !important;
        color: #24342D !important;
        -webkit-text-fill-color: #24342D !important;
        border: 1px solid #C8D0C1 !important;
        border-radius: 11px !important;
        caret-color: #173F35 !important;
    }

    .stTextInput input::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: #737C70 !important;
        -webkit-text-fill-color: #737C70 !important;
        opacity: 1 !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #173F35 !important;
        box-shadow: 0 0 0 2px rgba(23, 63, 53, 0.10) !important;
    }

    /* METRICS */

    [data-testid="stMetric"] {
        background: #FFFCF5 !important;
        border: 1px solid #DCDCCF;
        border-radius: 15px;
        padding: 17px 18px;
        box-shadow: 0 3px 10px rgba(23, 63, 53, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #737C70 !important;
    }

    [data-testid="stMetricValue"] {
        color: #173F35 !important;
        font-weight: 700;
    }

    /* FILE UPLOADER */

    [data-testid="stFileUploader"] {
        background: #FFFCF5 !important;
        border: 1px dashed #AABBA6;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    /* EXPANDERS AND FORMS */

    [data-testid="stExpander"] {
        background: #FFFCF5 !important;
        border: 1px solid #DCDCCF;
        border-radius: 13px;
    }

    [data-testid="stExpander"] summary {
        color: #173F35 !important;
        font-weight: 600;
    }

    [data-testid="stForm"] {
        background: #F1F0E5 !important;
        border: 1px solid #DCDCCF;
        border-radius: 14px;
        padding: 16px;
    }

    hr {
        border-color: #DCDCCF !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCDCCF;
        border-radius: 12px;
        overflow: hidden;
    }

    a {
        color: #173F35 !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 640px) {
        .main .block-container {
            padding: 1.2rem 0.9rem 2rem;
        }

        .brand {
            font-size: 29px;
        }

        [data-testid="stChatMessage"] {
            padding: 0.8rem;
            border-radius: 14px;
        }

        [data-testid="stMetric"] {
            padding: 12px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- SESSION STATE --------------------

defaults = {
    "messages": [],
    "documents": [],
    "evaluation": [],
    "uploaded_names": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------- DOCUMENT PROCESSING --------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_'-]+", text.lower())


def split_into_chunks(text, filename):
    words = clean_text(text).split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        content = " ".join(words[start:end])

        if content:
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
    try:
        return clean_text(
            uploaded_file.getvalue().decode("utf-8-sig")
        )
    except UnicodeDecodeError:
        return ""


def build_knowledge_base(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

        if extension not in {"txt", "md"}:
            continue

        content = read_uploaded_file(uploaded_file)

        if content:
            documents.extend(
                split_into_chunks(content, uploaded_file.name)
            )

    return documents


# -------------------- TF-IDF SEARCH --------------------

def build_search_index(chunks):
    token_lists = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    document_count = len(chunks)
    document_frequency = Counter()

    for tokens in token_lists:
        document_frequency.update(set(tokens))

    idf = {
        word: math.log(
            1 + document_count / (1 + count)
        ) + 1
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

        length = math.sqrt(
            sum(value * value for value in vector.values())
        )

        if length:
            vector = {
                word: value / length
                for word, value in vector.items()
            }

        vectors.append(vector)

    return idf, vectors


def search_documents(question, chunks, idf, vectors):
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
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "best_score": round(best_score, 4),
    }

    try:
        with open(
            "unresolved_questions.jsonl",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    except OSError:
        pass


# -------------------- GEMINI AI --------------------

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def generate_answer(question, history, ranked):
    api_key = get_api_key()

    if not api_key:
        return (
            "The AI service is not configured. Please add "
            "GEMINI_API_KEY in your Streamlit Secrets settings."
        ), []

    best_score = ranked[0][0] if ranked else 0.0

    useful_results = [
        (score, chunk)
        for score, chunk in ranked
        if score >= MIN_SCORE
    ]

    evidence = [
        {"score": score, "chunk": chunk}
        for score, chunk in useful_results
    ]

    evidence_text = "\n\n".join(
        f"Source filename: {item['chunk']['source']}\n"
        f"Passage: {item['chunk']['text'][:1200]}"
        for item in evidence
    )

    if ranked and not evidence:
        log_unresolved(question, best_score)

    if evidence_text:
        document_instruction = """
Relevant document passages are provided below.
Use them when the user asks about uploaded documents.
Refer to the source filename when using these passages.
Do not claim that a passage says something it does not say.
If the passages do not answer a document-specific question,
say that the available documents do not contain enough evidence.
You may still answer general questions using your general knowledge.
"""
    else:
        document_instruction = """
No sufficiently relevant document passages were found.
Answer general questions using your general knowledge.
If the user specifically asks what an uploaded document says,
explain that you could not find sufficient supporting evidence.
Do not invent document citations.
"""

    system_instruction = f"""
You are {APP_NAME}, a helpful, professional AI assistant.

Communication style:
- Be clear, natural, accurate and conversational.
- Give direct answers before extra explanation.
- Explain difficult ideas in simple words when appropriate.
- Use headings and lists only when they improve readability.
- Remember recent conversation to understand follow-up questions.
- Admit uncertainty instead of inventing facts.
- Do not claim to have browsed the internet or performed actions
  you did not actually perform.

Document instructions:
{document_instruction}

Retrieved document evidence:
{evidence_text if evidence_text else "No relevant evidence available."}
"""

    conversation_lines = []

    for message in history[-MAX_HISTORY_MESSAGES:]:
        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        conversation_lines.append(
            f"{role}: {message['content']}"
        )

    conversation_text = "\n".join(conversation_lines)

    prompt = f"""
{system_instruction}

Recent conversation:
{conversation_text}

Current user message:
{question}

Respond to the current message, using previous messages
to understand references and follow-up questions.
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = (
                "I couldn't generate a response this time. "
                "Please try asking again."
            )

        return answer, evidence

    except Exception as error:
        error_text = str(error).lower()

        if any(word in error_text for word in (
            "429", "quota", "rate limit", "resource_exhausted"
        )):
            message = (
                "The AI service has reached a usage limit. Please wait "
                "and try again, or check your Gemini API quota."
            )

        elif any(word in error_text for word in (
            "401", "403", "api key", "permission denied"
        )):
            message = (
                "The AI service could not authenticate. Check that your "
                "Gemini API key is valid and configured correctly "
                "in Streamlit Secrets."
            )

        elif "404" in error_text or "not found" in error_text:
            message = (
                "The selected AI model may not be available for your "
                "API key. Check model availability in Google AI Studio."
            )

        else:
            message = (
                "Something went wrong while contacting the AI service. "
                "Please try again later."
            )

        return message, evidence


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ◎ Omnis AI")
    st.caption("Your AI workspace")
    st.divider()

    st.markdown("#### Conversation")

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("#### Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="Upload text or Markdown files for document-based answers.",
    )

    if st.button("Load documents", use_container_width=True):
        new_names = [
            file.name
            for file in (uploaded_files or [])
        ]

        if not uploaded_files:
            st.warning("Select at least one TXT or Markdown file.")

        else:
            documents = build_knowledge_base(uploaded_files)

            if documents:
                st.session_state.documents = documents
                st.session_state.uploaded_names = new_names
                st.session_state.messages = []

                st.success(
                    f"Loaded {len(documents)} text sections."
                )

            else:
                st.warning(
                    "No readable text was found in those files."
                )

    if st.button("Load built-in sample", use_container_width=True):
        sample = """
        Omnis AI is a conversational document assistant.
        It divides documents into smaller overlapping chunks.
        TF-IDF retrieval ranks document sections against user questions.
        Relevant source passages can be displayed as evidence.
        The AI engine can answer general questions and explain concepts.
        Weak document evidence should not be presented as a verified fact.
        Retrieval quality can be tested using known questions and sources.
        Unanswered questions can be logged for future improvement.
        """

        st.session_state.documents = split_into_chunks(
            sample,
            "Built-in sample",
        )

        st.session_state.uploaded_names = ["Built-in sample"]
        st.session_state.messages = []

        st.success("Sample knowledge base loaded.")

    st.caption(
        f"{len(st.session_state.documents)} document sections loaded"
    )

    st.divider()
    st.caption("Supported files: TXT and Markdown")
    st.caption("AI answers require a configured Gemini API key.")


# -------------------- MAIN CHAT SCREEN --------------------

st.markdown(
    '<div class="brand">Omnis AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'A thoughtful space to ask, discover, and understand.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

metric1, metric2 = st.columns(2)

with metric1:
    st.metric(
        "Documents",
        len(st.session_state.documents),
    )

with metric2:
    st.metric(
        "Messages",
        sum(
            1 for item in st.session_state.messages
            if item["role"] == "user"
        ),
    )

if not st.session_state.messages:
    st.write("")
    st.markdown("### How can I help?")

    st.caption(
        "Ask a general question, explore an idea, or ask about "
        "your uploaded documents."
    )

    examples = [
        "Explain a difficult topic simply",
        "Help me write a professional email",
        "What can you find in my documents?",
    ]

    cols = st.columns(3)

    for index, example in enumerate(examples):
        with cols[index]:
            if st.button(
                example,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = example
                st.rerun()


# Display previous messages and source evidence.

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("evidence"):
            with st.expander("View document sources"):
                for item in message["evidence"]:
                    chunk = item["chunk"]

                    st.markdown(
                        f"**{chunk['source']}** · "
                        f"Similarity: {item['score']:.3f}"
                    )

                    st.write(chunk["text"])
                    st.divider()


# Process chat input.

pending_question = st.session_state.pop(
    "pending_question",
    "",
)

question = st.chat_input("Ask anything...")

if pending_question and not question:
    question = pending_question

if question and question.strip():
    question = question.strip()

    ranked = []

    if st.session_state.documents:
        idf, vectors = build_search_index(
            st.session_state.documents
        )

        ranked = search_documents(
            question,
            st.session_state.documents,
            idf,
            vectors,
        )

    history = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.spinner("Thinking..."):
        answer, evidence = generate_answer(
            question,
            history,
            ranked,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": evidence,
    })

    st.rerun()


# -------------------- RETRIEVAL EVALUATION --------------------

st.divider()

with st.expander("Document retrieval evaluation"):
    st.write(
        "Test whether the retriever finds the expected source "
        "for a known question."
    )

    with st.form("evaluation_form"):
        eval_question = st.text_input(
            "Test question",
            placeholder="What happens when document evidence is weak?",
        )

        expected_source = st.text_input(
            "Expected source filename",
            placeholder="Built-in sample",
        )

        submitted = st.form_submit_button("Run evaluation")

    if submitted:
        if (
            not eval_question.strip()
            or not expected_source.strip()
        ):
            st.warning(
                "Enter both a question and the expected source."
            )

        elif not st.session_state.documents:
            st.warning(
                "Load documents before running an evaluation."
            )

        else:
            idf, vectors = build_search_index(
                st.session_state.documents
            )

            results = search_documents(
                eval_question,
                st.session_state.documents,
                idf,
                vectors,
            )

            retrieved_source = (
                results[0][1]["source"]
                if results
                else "None"
            )

            passed = (
                retrieved_source.lower()
                == expected_source.strip().lower()
            )

            result = {
                "question": eval_question,
                "expected_source": expected_source.strip(),
                "retrieved_source": retrieved_source,
                "passed": passed,
            }

            st.session_state.evaluation.append(result)

            if passed:
                st.success(
                    f"PASS — retrieved {retrieved_source}"
                )

            else:
                st.warning(
                    f"Not a match — retrieved {retrieved_source}. "
                    "Check the source name or document content."
                )

    if st.session_state.evaluation:
        st.markdown("#### Evaluation history")

        st.dataframe(
            st.session_state.evaluation,
            use_container_width=True,
            hide_index=True,
        )

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


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "Omnis AI · Conversational AI and evidence-based retrieval"
)

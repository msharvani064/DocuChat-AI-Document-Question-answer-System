import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# TITLE
# -----------------------------

st.title("📄 DocuChat")

st.subheader(
    "AI-Based Document Question Answer System"
)

st.write(
    "Upload a PDF and ask questions about its content."
)


# -----------------------------
# LOAD HUGGING FACE MODEL
# -----------------------------

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


model = load_model()


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_text(pdf):

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# SPLIT DOCUMENT
# -----------------------------

def split_text(text):

    words = text.split()

    chunks = []

    chunk_size = 300

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


# -----------------------------
# FIND RELEVANT TEXT
# -----------------------------

def search_document(
    chunks,
    question
):

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    similarity = np.dot(
        chunk_embeddings,
        question_embedding
    )

    top_indexes = np.argsort(
        similarity
    )[-3:][::-1]

    results = []

    for index in top_indexes:

        results.append(
            chunks[index]
        )

    return results


# -----------------------------
# ASK OLLAMA
# -----------------------------

def ask_ai(
    question,
    context
):

    prompt = f"""
You are DocuChat, an AI document
question answering assistant.

Answer the question using ONLY
the information from the document.

If the answer is not present in
the document, say:

"I could not find the answer
in the uploaded document."

Document information:

{context}

Question:

{question}

Answer:
"""

    response = ollama.chat(

        model="llama3.2",

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
    ]


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("📚 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# -----------------------------
# MAIN APPLICATION
# -----------------------------

if uploaded_file:

    st.success(
        "PDF uploaded successfully!"
    )

    # Extract text

    with st.spinner(
        "Reading PDF..."
    ):

        text = extract_text(
            uploaded_file
        )

    if not text.strip():

        st.error(
            "Could not extract text from this PDF."
        )

    else:

        # Split text

        chunks = split_text(text)

        st.info(
            f"Document processed successfully. "
            f"{len(chunks)} text chunks created."
        )

        st.divider()

        # Question

        question = st.text_input(
            "💬 Ask a question about your document"
        )

        if question:

            # Search

            with st.spinner(
                "Searching document..."
            ):

                results = search_document(
                    chunks,
                    question
                )

            # Create context

            context = "\n\n".join(
                results
            )

            # AI answer

            with st.spinner(
                "Generating answer..."
            ):

                answer = ask_ai(
                    question,
                    context
                )

            # Display

            st.subheader(
                "🤖 AI Answer"
            )

            st.write(answer)

            # Sources

            with st.expander(
                "🔍 View relevant document content"
            ):

                for i, result in enumerate(
                    results,
                    1
                ):

                    st.write(
                        f"**Source {i}**"
                    )

                    st.write(result)

                    st.divider()

else:

    st.info(
        "👈 Upload a PDF from the sidebar to begin."
    )

    st.markdown(
        """
        ### How DocuChat works

        📄 **1. Upload PDF**

        🔍 **2. Find relevant information**

        🧠 **3. Hugging Face creates embeddings**

        🤖 **4. Ollama generates the answer**

        💬 **5. Answer is displayed**
        """
    )
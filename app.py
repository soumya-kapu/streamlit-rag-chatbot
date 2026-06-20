import streamlit as st
import os

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import chromadb

from llama_cpp import Llama


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Local Document Q&A using RAG")


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_0.gguf"

DB_PATH = "chroma_db"


# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_llm():

    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        verbose=False
    )


llm = load_llm()


# -----------------------------
# Load embeddings
# -----------------------------
@st.cache_resource
def load_embedding():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedder = load_embedding()


# -----------------------------
# Chroma DB
# -----------------------------
client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# PDF processing
# -----------------------------
def process_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)


    embeddings = embedder.encode(chunks)


    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()]
        )


    return len(chunks)



# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if uploaded_file:

    if st.button("Process Document"):

        count = process_pdf(uploaded_file)

        st.success(
            f"Processed {count} chunks"
        )



# -----------------------------
# Question Answer
# -----------------------------
question = st.text_input(
    "Ask a question"
)


if question:

    q_embedding = embedder.encode(
        [question]
    )[0]


    results = collection.query(
        query_embeddings=[
            q_embedding.tolist()
        ],
        n_results=3
    )


    context = "\n".join(
        results["documents"][0]
    )


    prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{question}

Answer:
"""


    response = llm(
        prompt,
        max_tokens=300
    )


    answer = response["choices"][0]["text"]


    st.write(answer)

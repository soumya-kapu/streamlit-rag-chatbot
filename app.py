import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import ollama

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")

client = chromadb.PersistentClient(path="./vectorstore")

collection = client.get_or_create_collection(
    name="pdf_collection"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = []

    for chunk in chunks:
        result = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(result["embedding"])

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    st.success(
        f"Stored {len(chunks)} chunks"
    )


question = st.chat_input(
    "Ask something about your PDF"
)


if question:

    q_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]


    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )


    context = "\n\n".join(
        result["documents"][0]
    )


    prompt = f"""
Answer only from this context:

{context}

Question:
{question}
"""


    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )


    st.write(
        response["message"]["content"]
    )

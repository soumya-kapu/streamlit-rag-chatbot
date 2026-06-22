import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Document Q&A App", page_icon="📄")

st.title("📄 Mozilla / RAG Document Q&A Demo")
st.write("Upload a PDF and ask questions about its content.")

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    return chunks

def retrieve_relevant_chunks(question, chunks, top_k=3):
    question_words = set(question.lower().split())
    scored = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored[:top_k] if score > 0]

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(uploaded_file)
            st.session_state.doc_chunks = chunks
        st.success(f"PDF processed successfully! {len(chunks)} chunks created.")

if st.session_state.doc_chunks:
    question = st.text_input("Ask a question about the uploaded document:")

    if question:
        relevant_chunks = retrieve_relevant_chunks(question, st.session_state.doc_chunks)

        if relevant_chunks:
            st.subheader("Answer")
            st.write("Here are the most relevant parts of the document:")

            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
        else:
            st.warning("No relevant answer found in the document.")
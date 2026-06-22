import streamlit as st
import re
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
        text += "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    return chunks

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def retrieve_relevant_chunks(question, chunks, top_k=3):
    question_words = set(re.findall(r'\w+', question.lower()))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(question_words.intersection(chunk_words))

        # extra boost if full question words appear multiple times
        for word in question_words:
            score += chunk.lower().count(word)

        scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, chunk in scored_chunks[:top_k]:
        if score > 0:
            results.append(clean_text(chunk))

    return results

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
            st.write(relevant_chunks[0])

            if len(relevant_chunks) > 1:
                with st.expander("See more relevant text"):
                    for i, chunk in enumerate(relevant_chunks[1:], 2):
                        st.markdown(f"**Relevant chunk {i}:**")
                        st.write(chunk)
        else:
            st.warning("Sorry, I could not find relevant information in the document.")
import streamlit as st
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Document Q&A App", page_icon="📄")

st.title("📄 Mozilla / RAG Document Q&A Demo")
st.write("Upload a PDF and ask questions about its content.")

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

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

    return text, chunks


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_section(question, text):
    q = question.lower()

    patterns = {
        "skills": [
            r"technical skills(.*?)(academic projects|projects|internships|training|personal information|declaration|$)"
        ],
        "projects": [
            r"academic projects(.*?)(internships|training|personal information|declaration|$)",
            r"projects(.*?)(internships|training|personal information|declaration|$)"
        ],
        "internships": [
            r"internships/training(.*?)(personal information|declaration|$)",
            r"internships(.*?)(personal information|declaration|$)",
            r"training(.*?)(personal information|declaration|$)"
        ],
        "education": [
            r"academic qualification(.*?)(technical skills|skills|projects|internships|training|$)",
            r"education(.*?)(technical skills|skills|projects|internships|training|$)"
        ],
        "objective": [
            r"career objective(.*?)(academic qualification|education|technical skills|skills|$)"
        ],
        "languages": [
            r"language known(.*?)(personal information|career objective|academic qualification|technical skills|$)",
            r"languages known(.*?)(personal information|career objective|academic qualification|technical skills|$)"
        ],
        "personal": [
            r"personal information(.*?)(career objective|academic qualification|technical skills|projects|declaration|$)"
        ]
    }

    if "skill" in q:
        section = "skills"
    elif "project" in q:
        section = "projects"
    elif "internship" in q or "training" in q:
        section = "internships"
    elif "education" in q or "qualification" in q:
        section = "education"
    elif "objective" in q or "career objective" in q:
        section = "objective"
    elif "language" in q:
        section = "languages"
    elif "personal" in q:
        section = "personal"
    else:
        return None

    for pattern in patterns[section]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip()
            result = clean_text(result)
            if result:
                return result

    return None


def retrieve_relevant_chunks(question, chunks, top_k=3):
    question_words = set(re.findall(r'\w+', question.lower()))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(question_words.intersection(chunk_words))

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
            text, chunks = process_pdf(uploaded_file)
            st.session_state.doc_text = text
            st.session_state.doc_chunks = chunks
        st.success(f"PDF processed successfully! {len(chunks)} chunks created.")


if st.session_state.doc_text:
    question = st.text_input("Ask a question about the uploaded document:")

    if question:
        # 1) try section extraction first
        section_answer = extract_section(question, st.session_state.doc_text)

        if section_answer:
            st.subheader("Answer")
            st.write(section_answer)

        else:
            # 2) fallback to generic retrieval
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
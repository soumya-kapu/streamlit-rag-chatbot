import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")


# ----------------------------
# Chat History
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Simple Embedding Replacement
# ----------------------------
documents = []


# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text


    if not text.strip():
        st.error("No text could be extracted from this PDF.")
        st.stop()


    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)


    # Store chunks in memory
    documents = chunks


    st.success(f"Stored {len(chunks)} chunks")


    # Preview chunks
    with st.expander("View First 3 Chunks"):

        for i, chunk in enumerate(chunks[:3]):
            st.markdown(f"### Chunk {i+1}")
            st.write(chunk)


    # Show previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    # Chat input

    question = st.chat_input(
        "Ask a question about the PDF..."
    )


    if question:


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):
            st.markdown(question)



        # Simple retrieval
        retrieved_chunks = documents[:3]


        context = "\n\n".join(
            retrieved_chunks
        )


        prompt = f"""
You are a helpful AI assistant.

Answer only using the context below.

If the answer is not available, say:
"I could not find that information in the document."

Context:

{context}


Question:

{question}
"""


        with st.spinner("Generating answer..."):

            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            answer = response["message"]["content"]



        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        with st.chat_message("assistant"):
            st.markdown(answer)



        with st.expander("📄 Retrieved Source Chunks"):

            for i, chunk in enumerate(retrieved_chunks):

                st.markdown(f"### Source {i+1}")
                st.write(chunk)

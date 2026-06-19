import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq


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
# Session Storage
# ----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Groq Client
# ----------------------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text


    if not text.strip():

        st.error(
            "Could not extract text from PDF"
        )

        st.stop()


    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_text(text)


    st.session_state.chunks = chunks


    st.success(
        f"Created {len(chunks)} chunks"
    )


    with st.expander(
        "View sample chunks"
    ):

        for i, chunk in enumerate(chunks[:3]):

            st.write(
                f"Chunk {i+1}"
            )

            st.write(chunk)



# ----------------------------
# Chat History
# ----------------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# ----------------------------
# Ask Question
# ----------------------------
if st.session_state.chunks:

    question = st.chat_input(
        "Ask something about your PDF..."
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
        relevant_chunks = []

        question_words = set(
            question.lower().split()
        )


        for chunk in st.session_state.chunks:

            chunk_words = set(
                chunk.lower().split()
            )

            score = len(
                question_words.intersection(
                    chunk_words
                )
            )


            if score > 0:
                relevant_chunks.append(chunk)



        if not relevant_chunks:

            relevant_chunks = (
                st.session_state.chunks[:3]
            )


        context = "\n\n".join(
            relevant_chunks[:3]
        )


        prompt = f"""
You are a helpful document assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
"I could not find this information in the document."

Context:

{context}


Question:

{question}
"""


        # LLM Response
        with st.spinner(
            "Generating answer..."
        ):

            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[

                    {
                        "role": "system",
                        "content":
                        "You answer questions from documents."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ]
            )


            answer = (
                response
                .choices[0]
                .message
                .content
            )



        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        with st.chat_message(
            "assistant"
        ):

            st.markdown(answer)



        with st.expander(
            "📄 Sources used"
        ):

            for i, chunk in enumerate(
                relevant_chunks[:3]
            ):

                st.write(
                    f"Source {i+1}"
                )

                st.write(chunk)

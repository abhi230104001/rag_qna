

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file.")
    st.stop()


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("./doc_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3


# ============================================================
# STREAMLIT SESSION STATE
# ============================================================

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_document(path):

    # Load PDFs
    loader = PyPDFDirectoryLoader(str(path))
    docs = loader.load()

    if not docs:
        raise ValueError("No readable PDF documents found.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text chunks could be extracted.")

    # Create Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # Create in-memory vector store
    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    st.session_state.vector_store = vector_db

    # Initialize Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    # ========================================================
    # RETRIEVAL TOOL
    # ========================================================

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant information from uploaded PDF documents."""

        retrieved_docs = vector_db.similarity_search(
            query=query,
            k=TOP_K,
        )

        if not retrieved_docs:
            return "No relevant information was found."

        # Combine all retrieved chunks
        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        return context

    # ========================================================
    # AGENT CONFIGURATION
    # ========================================================

    system_prompt = """
    You are a helpful PDF question-answering assistant.

    Your knowledge base consists of uploaded PDF documents.

    ALWAYS use the retrieve_context tool for questions
    requiring information from uploaded documents.

    Answer using the retrieved context.
    If the answer is not present in the context,
    clearly say that the uploaded documents do not
    contain the answer.

    Do not invent information.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ============================================================
# EXTRACT PLAIN TEXT FROM MODEL RESPONSE
# ============================================================

def extract_answer(response):

    content = response["messages"][-1].content

    # Gemini may return content as a list of blocks
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    return str(content)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Gemini PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Gemini PDF RAG Chatbot")

st.caption(
    "Upload PDFs and ask questions about their contents."
)


# ============================================================
# PDF UPLOAD UI
# ============================================================

if not st.session_state.document_uploaded:

    uploaded_files = st.file_uploader(
        label="Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        with st.spinner("Processing documents..."):

            try:

                # Save uploaded PDFs
                for file in uploaded_files:

                    target_path = UPLOAD_DIR / file.name

                    target_path.write_bytes(
                        file.getvalue()
                    )

                # Process and index documents
                process_document(UPLOAD_DIR)

                st.success("Documents processed successfully!")

                st.rerun()

            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )


# ============================================================
# CHAT UI
# ============================================================

if (
    st.session_state.document_uploaded
    and st.session_state.agent
):

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    query = st.chat_input(
        "Ask anything about your uploaded documents..."
    )

    if query:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Invoke agent
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": query,
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": "pdf-chat",
                            }
                        },
                    )

                    # Extract only plain text
                    answer = extract_answer(response)

                    st.markdown(answer)

                    # Save clean answer
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.error(f"Chat failed: {e}")

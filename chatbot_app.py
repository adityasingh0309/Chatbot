import streamlit as st
import ollama
import pandas as pd
import PyPDF2
import json

# -------------------------------
# Helper: Text Chunking Function
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split long text into chunks with overlap.
    chunk_size: number of characters per chunk
    overlap: number of overlapping characters between chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        # Move start forward with overlap
        start = end - overlap

        if start < 0:
            start = 0

    return [c for c in chunks if c]  # remove empty chunks


# Page Configuration
st.set_page_config(page_title="🤖 Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Local Ollama Chatbot")

# Sidebar for Chat History & Model Selection
st.sidebar.title("💬 Chat History")

# Model selector (from your installed models)
available_models = [
    "phi3:latest",
    "mistral:latest",
    "gemma2:2b",
    "llama3.1:latest"
]
model_name = st.sidebar.selectbox("🧠 Choose AI Model", available_models, index=0)

# -------------------------------
# File Uploader Section
# -------------------------------
st.sidebar.subheader("📂 Upload a File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (PDF, TXT, or CSV)", 
    type=["pdf", "txt", "csv"]
)

# Initialize session_state for doc & chunks
if "doc_json" not in st.session_state:
    st.session_state.doc_json = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

file_content = ""

if uploaded_file is not None:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    # -------- Extract File Content --------
    if uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8")
        st.sidebar.text_area("File Preview (Text)", file_content[:400], height=150)

    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            file_content += page.extract_text() or ""
        st.sidebar.text_area("File Preview (PDF)", file_content[:400], height=150)

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.sidebar.dataframe(df.head())
        # You can decide how to convert CSV into text; here we use head as text
        file_content = df.to_csv(index=False)

    # -------- Store as JSON --------
    st.session_state.doc_json = {
        "filename": uploaded_file.name,
        "type": uploaded_file.type,
        "content": file_content
    }

    # -------- Chunking Process --------
    if file_content.strip():
        raw_chunks = chunk_text(file_content, chunk_size=800, overlap=150)
        st.session_state.chunks = [
            {"id": i, "text": chunk}
            for i, chunk in enumerate(raw_chunks)
        ]

        # Show basic info
        st.sidebar.info(f"Document split into {len(st.session_state.chunks)} chunks.")

        # Optional: show JSON view of chunks (trimmed)
        with st.sidebar.expander("📄 Chunk JSON Preview"):
            # Only show first few chunks to avoid huge JSON
            preview_chunks = st.session_state.chunks[:3]
            st.json(preview_chunks)

# -------------------------------
# Chat Session Management
# -------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current" not in st.session_state:
    st.session_state.current = None

# Create a New Chat Session
if st.sidebar.button("➕ New Chat"):
    new_chat = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat] = []
    st.session_state.current = new_chat

# Select a Chat Session
if st.session_state.chats:
    selected = st.sidebar.radio(
        "Select Chat",
        list(st.session_state.chats.keys()),
        index=list(st.session_state.chats.keys()).index(st.session_state.current)
        if st.session_state.current else 0
    )
    st.session_state.current = selected
else:
    st.sidebar.info("Click 'New Chat' to start one.")

# -------------------------------
# Build Context from Chunks
# -------------------------------
def build_context_from_chunks(max_chars=2000):
    """
    Combine chunk texts into a single context string,
    limited by max_chars.
    """
    if not st.session_state.chunks:
        return ""

    combined = ""
    for ch in st.session_state.chunks:
        text = ch["text"]
        if len(combined) + len(text) + 1 > max_chars:
            break
        combined += text + "\n"
    return combined.strip()


# Chat Input Box
user_input = st.chat_input("Type your message...")

# Process Chat Input
if user_input and st.session_state.current:

    # Use chunked document as context instead of just first 200 chars
    context_text = build_context_from_chunks(max_chars=2000)

    if context_text:
        final_prompt = f"""
You are a helpful assistant. The user has uploaded a document.

Use the following document chunks as context to answer the user's question.
If the answer is not in the document, say so clearly.

User Question:
{user_input}

Document Chunks:
{context_text}
"""
    else:
        final_prompt = user_input

    # Send prompt to selected Ollama model
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": final_prompt}]
    )
    bot_reply = response['message']['content']

    # Save Messages
    st.session_state.chats[st.session_state.current].append(("You", user_input))
    st.session_state.chats[st.session_state.current].append(("Bot", bot_reply))

# Display Chat Messages
if st.session_state.current:
    for role, msg in st.session_state.chats[st.session_state.current]:
        st.chat_message("user" if role == "You" else "assistant").markdown(msg)

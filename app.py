GOOGLE_API_KEY = "AIzaSyACugEZqeqCwyDK2IcW_LjoF7flGsHeyiA"

import os
import streamlit as st
import faiss
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --- Config Gemini ---
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel("models/gemini-2.0-flash")

# --- Embed model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- PDF to Text ---
def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Chunk Text ---
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Create FAISS Index ---
def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# --- Search Index ---
def search_index(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# --- Generate Answer ---
def get_gemini_answer(question, context):
    prompt = f"""Answer the question based on the context below. Be accurate and specific.

Context:
{context}

Question:
{question}
"""
    response = llm_model.generate_content(prompt)
    return response.text

# --- Streamlit Setup ---
st.set_page_config(page_title="Smart PDF Chat", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2);
            background-size: cover;
            background-position: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .stTextInput > div > input {
            background-color: #f0f2f6;
            color: black;
            font-size: 16px;
        }
        .css-1cpxqw2 {
            background-color: rgba(255,255,255,0.8) !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Smart PDF Chat Assistant")

# --- Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        full_text = pdf_to_text(uploaded_file)
        chunks = chunk_text(full_text)
        index, embeddings, chunk_data = create_faiss_index(chunks)
        st.success("PDF processed and ready!")

    # Chat Input
    user_input = st.chat_input("Ask something about the uploaded PDF")

    if user_input:
        relevant_chunks = search_index(user_input, index, chunk_data)
        context = "\n\n".join(relevant_chunks)
        answer = get_gemini_answer(user_input, context)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

    # Display History
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(f"**You:** {msg}")
        else:
            st.chat_message("assistant").markdown(f"**Gemini:** {msg}")

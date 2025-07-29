GOOGLE_API_KEY="AIzaSyACugEZqeqCwyDK2IcW_LjoF7flGsHeyiA"

import os
import streamlit as st
import faiss
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
genai.configure(api_key=GOOGLE_API_KEY)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini model
llm_model = genai.GenerativeModel("models/gemini-2.0-flash")

# --- PDF to Text ---
def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Text Chunking ---
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Embed and Store ---
def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# --- Search Relevant Chunks ---
def search_index(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# --- Gemini QA ---
def get_gemini_answer(question, context):
    prompt = f"""
Answer the question based on the context below. Be accurate and specific.

Context:
{context}

Question:
{question}
"""
    response = llm_model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A with Gemini", layout="centered")
st.title("📄 Your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        full_text = pdf_to_text(uploaded_file)
        chunks = chunk_text(full_text)
        index, embeddings, chunk_data = create_faiss_index(chunks)
        st.success("PDF processed successfully!")

    # Chat interface
    user_input = st.chat_input("Ask a question about the PDF")

    if user_input:
        relevant_chunks = search_index(user_input, index, chunk_data)
        context = "\n\n".join(relevant_chunks)
        answer = get_gemini_answer(user_input, context)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

    # Display conversation
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)

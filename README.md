# PDF Question Answering Web App

This is a Streamlit-based web application that allows users to upload a PDF document and ask questions about its content. The system uses the Retrieval-Augmented Generation (RAG) approach to extract relevant information and generate accurate answers using Google's Gemini AI model.

## Features

- Upload and read large PDF files (including 500+ pages)
- Extract and chunk PDF content into manageable text segments
- Generate vector embeddings using Sentence Transformers
- Create a FAISS index for efficient similarity search
- Ask questions in a chat interface
- Retrieve relevant context from PDF and generate answers using Gemini LLM
- Interactive and responsive user interface

## Technologies Used

- Python
- Streamlit
- Google Generative AI (Gemini)
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers
- PyPDF
- HTML/CSS (for custom styling)

## Architechure

- PDF → Extracted Text → Chunking → Sentence Embeddings
- Embeddings → FAISS Index → Similarity Search
- Retrieved Context + User Query → Prompt to Gemini API
- Gemini Response → Displayed in Chat Interface

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/maheshsaid76/AI-Assignment.git
cd pdf-qa-gemini

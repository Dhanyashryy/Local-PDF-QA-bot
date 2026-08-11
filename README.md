# Local AI PDF Q&A Bot

A local Retrieval-Augmented Generation (RAG) chatbot built with Ollama, LangChain, and Chroma.  
It answers questions based on uploaded PDF documents and runs locally.

## Features

- Upload one or more PDF files
- Split PDFs into chunks
- Create embeddings with Ollama
- Store embeddings in a local Chroma database
- Ask questions about the uploaded documents
- Option to show retrieved context

## Requirements

- Python 3.10+
- Ollama installed locally
- Models pulled:
  - `llama3.2`
  - `nomic-embed-text`

## Setup

1. Install Ollama and pull the models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## How it works

1. Upload one or more PDF files.
2. Click **Process PDFs**.
3. Ask questions about the content.
4. The app retrieves relevant chunks and sends them to Ollama for a grounded answer.

## Notes

- Works best with text-based PDFs.
- Scanned PDFs may need OCR.
- The Chroma database is stored locally in `chroma_db/`.

## Sample Questions

- What is this document about?
- Explain the main points.
- What does the document say about [topic]?

## Project Stack

- Ollama
- LangChain
- Chroma
- Streamlit
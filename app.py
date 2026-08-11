import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

st.set_page_config(page_title="Local PDF Q&A Bot", layout="wide")
st.title("Local AI PDF Q&A Bot")

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_bot"

def load_and_process_pdfs(uploaded_files):
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    return chunks, all_docs

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

show_context = st.checkbox("Show retrieved context", value=True)

if uploaded_files:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if st.button("Process PDFs"):
        with st.spinner("Loading, splitting, and embedding PDFs..."):
            chunks, docs = load_and_process_pdfs(uploaded_files)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            # Delete the previous collection before creating a fresh one
            try:
                old_vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIR
                )
                old_vector_store.delete_collection()
            except Exception:
                # The collection may not exist the first time
                pass

            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR
            )

        st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks.")

if st.session_state.get("vector_store") is not None:
    question = st.text_input("Ask a question about the uploaded PDFs:")

    if question:
        results = st.session_state.vector_store.similarity_search(question, k=3)

        if show_context:
            st.subheader("Retrieved Context")
            for i, doc in enumerate(results, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)
                st.write("---")

        context = "\n\n".join(doc.page_content for doc in results)
        llm = ChatOllama(model="llama3.2")

        prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know.
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)
else:
    st.info("Upload PDFs and click **Process PDFs** to build the knowledge base.")
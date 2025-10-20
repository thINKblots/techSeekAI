# build_index.py

import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
# 1. Path to the directory containing your PDF files
PDFS_PATH = "OEM_manuals/" 

# 2. Path where you want to save the FAISS index
INDEX_PATH = "faiss_index"

# --- Main Indexing Logic ---
def create_index():
    """
    This function builds a FAISS vector store from PDF documents
    and saves it to disk.
    """
    print("Starting the indexing process...")

    # Load all PDF files from the specified path
    all_files = glob.glob(os.path.join(PDFS_PATH, "*.pdf"))
    if not all_files:
        print(f"No PDF files found in '{PDFS_PATH}'. Please check the path.")
        return
        
    print(f"Found {len(all_files)} PDF files to process.")

    # Load the documents
    documents = []
    for file_path in all_files:
        try:
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Successfully loaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not documents:
        print("Could not load any documents. Exiting.")
        return

    # Split the documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Initialize the embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store from the chunks
    print("Creating FAISS index from document chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS index to disk
    print(f"Saving FAISS index to '{INDEX_PATH}'...")
    vector_store.save_local(INDEX_PATH)

    print("✅ Indexing complete and saved successfully!")


if __name__ == "__main__":
    create_index()
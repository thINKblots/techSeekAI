# build_index.py

import os
# This should be at the top.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import glob
import multiprocessing # 👈 IMPORT THIS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
PDFS_PATH = "OEM_manuals/Skyjack/Service Manuals" 
INDEX_PATH = "chroma_index"

def main():
    """Main function to run the indexing process."""
    print("Starting the indexing process...")

    all_files = glob.glob(os.path.join(PDFS_PATH, "*.pdf"))
    if not all_files:
        print(f"No PDF files found in '{PDFS_PATH}'. Please check the path.")
        return
        
    print(f"Found {len(all_files)} PDF files to process.")

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )

    print(f"Creating and persisting ChromaDB index to '{INDEX_PATH}'...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=INDEX_PATH
    )

    print("✅ Indexing complete and saved successfully!")

# --- ❗️KEY CHANGE IS HERE ---
if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid multiprocessing deadlocks on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()
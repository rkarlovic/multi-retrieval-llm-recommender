import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredRTFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)

# --------------------------
# FILE LOADER BASED ON EXT
# --------------------------

def load_file(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        return TextLoader(path).load()

    elif ext == ".pdf":
        return PyPDFLoader(path).load()

    elif ext == ".docx":
        return Docx2txtLoader(path).load()

    elif ext == ".rtf":
        return UnstructuredRTFLoader(path).load()

    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(path).load()

    elif ext == ".md":
        return UnstructuredMarkdownLoader(path).load()

    # fallback loader for anything else
    return UnstructuredFileLoader(path).load()


# --------------------------
# CHUNKING FUNCTION
# --------------------------

def chunk_documents(documents, chunk_size=400, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# --------------------------
# LOAD ALL FILES FROM FOLDER
# --------------------------

def load_and_chunk_folder(folder_path: str):
    documents = []

    print(f"\n📁 Scanning folder: {folder_path}\n")

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        print(f"Loading: {filename}")

        try:
            docs = load_file(filepath)
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")

    print(f"\nLoaded {len(documents)} documents. Chunking...\n")

    chunks = chunk_documents(documents)

    print(f"Generated {len(chunks)} chunks.\n")
    return chunks


# --------------------------
# SAVE CHUNKS TO JSON
# --------------------------

def save_chunks_to_json(chunks, output_path="chunks.json"):
    data = []
    for chunk in chunks:
        data.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved {len(chunks)} chunks to {output_path}")


# --------------------------
# MAIN ENTRY POINT
# --------------------------

if __name__ == "__main__":
    folder_path = "corpus"   # <-- put your folder name here
    output_path = "chunks.json"

    chunks = load_and_chunk_folder(folder_path)
    save_chunks_to_json(chunks, output_path)

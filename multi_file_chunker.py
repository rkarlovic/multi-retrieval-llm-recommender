import os
from pathlib import Path
from typing import List, Dict


class MultiFileChunker:
    """
    Multi-file chunking module compliant with the Development Document.
    
    RULE:
    - Each file represents EXACTLY ONE chunk.
    - No splitting.
    - Chunk text = full file content.
    """

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

    def load_text(self, file_path: Path) -> str:
        """
        Reads the raw text from a file.
        """
        return file_path.read_text(encoding="utf-8")

    def process_files(self) -> List[Dict]:
        """
        Processes every text file in the folder.
        Each file becomes exactly one chunk.
        """
        chunks = []
        file_list = sorted(self.folder_path.glob("*.*"))  # process all files

        for idx, file_path in enumerate(file_list, start=1):
            raw_text = self.load_text(file_path)

            chunk = {
                "id": f"{file_path.stem}_chunk_1",   # one chunk per file
                "document_id": file_path.stem,        # document name = file name without extension
                "chunk_index": 1,                     # always 1
                "text": raw_text
            }

            chunks.append(chunk)

        return chunks


# Example usage
if __name__ == "__main__":
    folder = "documents"  # folder where each document is stored

    chunker = MultiFileChunker(folder)
    output_chunks = chunker.process_files()

    for chunk in output_chunks:
        print("Chunk ID:", chunk["id"])
        print("Document ID:", chunk["document_id"])
        print("Text Preview:", chunk["text"][:100], "...")
        print("-----")

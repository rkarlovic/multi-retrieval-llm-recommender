import os
import json
from pathlib import Path
from typing import List, Dict


class MultiFileChunker:
    """
    Multi-file chunking module.
    RULE:
    - Each file = one chunk.
    """

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

    # ------------------------------
    # Load full file content
    # ------------------------------
    def load_text(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

    # ------------------------------
    # Process all files → chunks
    # ------------------------------
    def process_files(self) -> List[Dict]:
        chunks = []
        file_list = sorted(self.folder_path.glob("*.*"))

        for file_path in file_list:
            raw_text = self.load_text(file_path)

            chunk = {
                "id": f"{file_path.stem}_chunk_1",
                "document_id": file_path.stem,
                "chunk_index": 1,
                "text": raw_text
            }

            chunks.append(chunk)

        return chunks

    # ------------------------------
    # Save all chunks → JSON file
    # ------------------------------
    def save_chunks_to_file(self, output_path: str, chunks: List[Dict]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # ------------------------------
    # Full pipeline
    # ------------------------------
    def run(self, output_path: str) -> List[Dict]:
        chunks = self.process_files()
        self.save_chunks_to_file(output_path, chunks)
        return chunks


# Example usage
if __name__ == "__main__":
    chunker = MultiFileChunker("corpus")

    chunks = chunker.run(output_path="chunk_output/all_chunks.json")

    print(f"Saved {len(chunks)} file-based chunks.")

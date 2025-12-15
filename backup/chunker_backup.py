import re
import json
from pathlib import Path
from typing import List, Dict

#Chunk = paragraph separated by blank lines
class Chunker:

    def __init__(self):
        pass

    def load_document(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        return path.read_text(encoding="utf-8")

    def split_into_paragraphs(self, text: str) -> List[str]:
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on blank lines
        paragraphs = re.split(r"\n\s*\n", text.strip())

        cleaned = [p.strip() for p in paragraphs if p.strip()]
        return cleaned

    
    def generate_chunks(self, paragraphs: List[str], document_id: str) -> List[Dict]:
        chunks = []
        for idx, paragraph in enumerate(paragraphs, start=1):
            chunk = {
                "id": f"{document_id}_chunk_{idx}",
                "document_id": document_id,
                "paragraph_index": idx,
                "text": paragraph
            }
            chunks.append(chunk)

        return chunks

    
    def save_chunks_to_file(self, output_path: str, chunks: List[Dict]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    
    def process_document(self, file_path: str, document_id: str, output_path: str) -> List[Dict]:
        raw_text = self.load_document(file_path)
        paragraphs = self.split_into_paragraphs(raw_text)
        chunks = self.generate_chunks(paragraphs, document_id)

        self.save_chunks_to_file(output_path, chunks)

        return chunks

if __name__ == "__main__":
    chunker = Chunker()

    chunks = chunker.process_document(
        file_path="corpus_merged.txt",
        document_id="destination_losinj",
        output_path="chunk_output/corpus_merged.json"
    )

    print(f"Saved {len(chunks)} chunks.")

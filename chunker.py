import re
from pathlib import Path
from typing import List, Dict

class Chunker:
    """
    Paragraph-based chunking module.
    Compliant with Development Document (Section 2.2 - Chunking Module).
    - Each chunk corresponds to one logical paragraph.
    - Paragraphs are defined as text blocks separated by blank lines.
    """

    def __init__(self):
        pass

    def load_document(self, file_path: str) -> str:
        """
        Loads raw text from a document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        return path.read_text(encoding="utf-8")

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Splits text into paragraphs.
        A paragraph is defined as ANY block of text separated by one or more blank lines.
        This rule follows Option A as chosen by the user.
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on one or more blank lines
        paragraphs = re.split(r"\n\s*\n", text.strip())

        # Trim internal whitespace
        cleaned = [p.strip() for p in paragraphs if p.strip()]

        return cleaned

    def generate_chunks(self, paragraphs: List[str], document_id: str) -> List[Dict]:
        """
        Converts paragraphs into structured chunks with metadata.
        Aligns with Development Document requirements:
        Each chunk stores:
        - text
        - metadata (document_id, paragraph_index)
        """
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

    def process_document(self, file_path: str, document_id: str) -> List[Dict]:
        """
        Loads → splits → chunks document into structured data.
        This is the main function your pipeline will call.
        """
        raw_text = load = self.load_document(file_path)
        paragraphs = self.split_into_paragraphs(raw_text)
        chunks = self.generate_chunks(paragraphs, document_id)
        return chunks


# Example usage:
if __name__ == "__main__":
    chunker = Chunker()

    # Replace with your file path:
    file_path = "corpus/corpus_merged.txt"

    chunks = chunker.process_document(file_path, document_id="destination_lošinj")

    for chunk in chunks:
        print(chunk["id"])
        print(chunk["text"])
        print("----")

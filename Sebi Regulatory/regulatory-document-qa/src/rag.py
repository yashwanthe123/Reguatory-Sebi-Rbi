"""Page-aware semantic retrieval for regulatory PDF documents."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable



@dataclass
class Passage:
    filename: str
    page: int
    text: str


def list_pdf_files(directory: str | Path) -> list[Path]:
    folder = Path(directory)
    return sorted(p for p in folder.rglob("*.pdf") if p.is_file()) if folder.exists() else []


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_words: int = 180, overlap_words: int = 35) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Use chunk_words > overlap_words >= 0.")
    chunks, step = [], chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_passages(pdf_paths: Iterable[str | Path], chunk_words=180, overlap_words=35) -> list[Passage]:
    passages = []
    import fitz
    for value in pdf_paths:
        path = Path(value)
        try:
            with fitz.open(path) as pdf:
                for page_no, page in enumerate(pdf, start=1):
                    text = clean_text(page.get_text("text"))
                    passages.extend(Passage(path.name, page_no, chunk)
                                    for chunk in chunk_text(text, chunk_words, overlap_words))
        except Exception as exc:
            raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
    return passages


class RegulatoryQA:
    def __init__(self, pdf_paths, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 chunk_words=180, overlap_words=35):
        self.passages = extract_passages(pdf_paths, chunk_words, overlap_words)
        if not self.passages:
            raise ValueError("No selectable PDF text found. Scanned PDFs need OCR first.")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self._np = np
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [p.text for p in self.passages],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        if not question.strip():
            raise ValueError("Enter a question.")
        query = self.model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self._np.dot(self.embeddings, query)
        indexes = self._np.argsort(-scores, kind="stable")[:max(1, top_k)]
        return [
            {"filename": self.passages[int(i)].filename,
             "page": self.passages[int(i)].page,
             "text": self.passages[int(i)].text,
             "score": float(scores[i])}
            for i in indexes
        ]

    def answer(self, question: str, top_k: int = 4, min_score: float = 0.25) -> dict:
        retrieved = self.retrieve(question, top_k)
        relevant = [item for item in retrieved if item["score"] >= min_score]
        if not relevant:
            answer = ("I couldn't find a sufficiently relevant passage in the loaded documents. "
                      "Try rephrasing the question or adding the relevant official circular or regulation.")
            sources = []
        else:
            lead = relevant[:3]
            answer = "Based on the retrieved passages:\n\n" + "\n\n".join(
                f"“{item['text']}” [{item['filename']}, p. {item['page']}]"
                for item in lead
            )
            answer += ("\n\nThis is a retrieval-based response, not a legal interpretation. "
                       "Check the cited pages and current official version before relying on it.")
            sources = [{"filename": x["filename"], "page": x["page"], "score": x["score"]}
                       for x in lead]
        return {"question": question, "answer": answer, "sources": sources, "retrieved": retrieved}

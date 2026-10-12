# Regulatory Desk — Document Q&A

A local question-answering tool for regulatory PDFs. Add SEBI, RBI, or other official PDF documents, ask a question, and inspect the passages retrieved from those documents with page-level citations.

This version focuses on the retrieval part of a RAG workflow: PDF extraction, page-aware chunking, sentence-transformer embeddings, similarity ranking, and evidence-first responses. It does not invent an LLM-generated legal interpretation; instead, it surfaces source passages with filenames and page numbers so they can be checked.

## Features

- Extract text from PDFs with PyMuPDF, preserving page numbers
- Split text into overlapping chunks
- Encode chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Retrieve relevant passages with cosine similarity
- Provide a fallback when no passage meets the relevance threshold
- Display passages and citations in Streamlit
- Download results as JSON
- CLI and unit tests for text chunking

## Requirements

- Python 3.10–3.12 recommended
- Internet access on first run to download the embedding model
- Text-based PDFs; scanned documents need OCR, which is not included

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place official PDFs in `data/regulations/`. Get current documents from the [SEBI website](https://www.sebi.gov.in/) or [RBI website](https://www.rbi.org.in/).

Start the app:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

CLI example:

```powershell
python -m src.cli "What does this document say about disclosure requirements?" --docs data/regulations --output answer.json
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

## How it works

1. Extract each PDF page separately.
2. Split page text into 180-word chunks with 35-word overlap.
3. Embed passages and the user's question.
4. Rank passages by cosine similarity.
5. Show passages above a relevance threshold with filename and page citations; otherwise return a fallback.

## Limitations

- This is retrieval-based Q&A, not a generative LLM that writes new answers.
- Similarity scores are ranking signals, not calibrated probabilities.
- Scanned PDFs require OCR.
- The app does not automatically fetch the latest rules; add current official PDFs yourself.
- AWS deployment is not included.
- This is a research aid, not legal, compliance, or investment advice.

## Next improvements

- Add OCR for scanned circulars
- Add an LLM that answers only from retrieved context
- Evaluate retrieval using questions with known source pages
- Filter by regulator, date, circular number, and topic
- Add document metadata and an update workflow
- Deploy with appropriate access controls

## Resume description

**Regulatory Document Q&A System | Python, NLP, Sentence Transformers, Streamlit**

- Built a page-aware PDF retrieval pipeline for regulatory documents using sentence-transformer embeddings.
- Implemented overlapping chunking, semantic similarity ranking, and page-level source references.
- Developed an interactive interface with relevance thresholds, evidence display, JSON export, and a no-relevant-source fallback.

Only claim features you have run and can explain. This baseline does not yet use an LLM to generate answers or include AWS deployment.

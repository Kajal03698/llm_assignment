# Architecture (short)

## Flow

```
data/raw (TXT, HTML, PDF)
    → ingest.py
    → rag.py (chunk + FAISS + Azure embeddings)
    → data/index

User question
    → query rewrite (LLM)
    → similarity search (FAISS)
    → rerank (embeddings)
    → guardrails
    → answer (LLM + citations)
```

## Choices

- **FAISS**: simple local vector store.
- **Similarity search**: straightforward; reranking improves order.
- **Query rewrite**: helps when users say "profits" but filings say "net income".
- **Guardrails**: block non-financial questions and weak retrieval.

## Limits

- Small sample corpus (not full Kaggle SEC data).
- PDF text extraction is basic (no OCR for scanned PDFs).
- Evaluation uses LLM-as-judge (extra API calls).

# Financial RAG Assistant

A question-answering system for SEC filings (10-K and 10-Q). Ask about revenue, profit margins, risks, anything in the documents—get back grounded answers with citations.

## How to use it

Get Python 3.9+ running. Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Azure OpenAI API key and deployment names.

Build the index (one-time):
```bash
python build_index.py
```

Run the web interface:
```bash
streamlit run app.py
```

Open http://localhost:8501. Type a question or click one of the demo questions in the sidebar.

## How it works

1. Load SEC filings as TXT
2. Split into chunks (800 tokens, 150 overlap)
3. Embed with Azure text-embedding-3-small, store in FAISS
4. When you ask a question:
   - Rewrite the query (catch synonym mismatches)
   - Similarity search in FAISS
   - Rerank by embedding cosine similarity
   - Pass top-5 chunks to gpt-4o-mini with instructions to cite sources
   - Check guardrails (financial question? sufficient context?)
5. Return answer with links to the filing excerpts

## Testing it

Run the evaluation suite on 22 test questions:
```bash
python evaluate.py
```

Results go to `evaluation/results/evaluation_results.json`.

## Files

- `ingest.py` — parse TXT/HTML/PDF files
- `rag.py` — chunking, embedding, retrieval, reranking, answer generation
- `app.py` — Streamlit frontend
- `config.py` — read from `.env`
- `data/raw/` — your filing documents
- `data/index/` — generated FAISS index

## Notes

- Requires Azure OpenAI (chat + embedding deployments)
- Chunking and retrieval settings in `.env` or hardcoded defaults
- See `ARCHITECTURE.md` for design details

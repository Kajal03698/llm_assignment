"""
RAG pipeline: chunk, index, retrieve (rewrite + rerank), answer with citations.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingest import Filing, load_all_files

# --- Azure models ---


def get_embeddings():
    if not config.AZURE_API_KEY or not config.AZURE_ENDPOINT:
        raise ValueError("Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
    return AzureOpenAIEmbeddings(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_deployment=config.EMBED_DEPLOYMENT,
    )


def get_chat():
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_deployment=config.CHAT_DEPLOYMENT,
        temperature=0.1,
    )


# --- Chunking ---


def make_chunks(filings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for f in filings:
        for section_name, text in f.sections.items():
            if len(text) < 100:
                continue
            header = f"[{f.company} | {f.form_type} | FY{f.fiscal_year} | {section_name}]\n"
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append(
                    Document(
                        page_content=header + piece,
                        metadata={
                            "company": f.company,
                            "ticker": f.ticker,
                            "form_type": f.form_type,
                            "fiscal_year": f.fiscal_year,
                            "section": section_name,
                            "source_path": f.source_path,
                            "chunk_id": f"{f.doc_id}_{i}",
                        },
                    )
                )
    return chunks


# --- Index ---


def build_index():
    print("Loading filings...")
    filings = load_all_files(config.DATA_DIR)
    print(f"  {len(filings)} files")
    chunks = make_chunks(filings)
    print(f"  {len(chunks)} chunks")
    print("Building FAISS index (calls Azure embeddings)...")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(config.INDEX_DIR))
    print("Done. Index saved to data/index/")


def load_index():
    path = config.INDEX_DIR / "index.faiss"
    if not path.exists():
        raise FileNotFoundError("Run: python build_index.py")
    return FAISS.load_local(
        str(config.INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


# --- Query rewrite ---


def rewrite_query(question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite this financial filing question into 2 search queries as a JSON array of strings only."),
        ("human", "{question}"),
    ])
    llm = get_chat()
    text = llm.invoke(prompt.format_messages(question=question)).content.strip()
    try:
        extra = json.loads(text)
        if isinstance(extra, list):
            return [question] + [str(q) for q in extra[:2]]
    except json.JSONDecodeError:
        pass
    return [question]


# --- Rerank ---


def rerank(question, docs, scores, top_n=5):
    if not docs:
        return [], []
    emb = get_embeddings()
    q_vec = np.array(emb.embed_query(question))
    d_vecs = np.array(emb.embed_documents([d.page_content for d in docs]))
    sims = (d_vecs @ q_vec) / (np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9)
    order = np.argsort(sims)[::-1][:top_n]
    return [docs[i] for i in order], [float(sims[i]) for i in order]


# --- Retrieve ---


def retrieve(store, question):
    top_k = config.TOP_K
    top_n = config.TOP_N_AFTER_RERANK

    queries = [question]
    if config.ENABLE_QUERY_REWRITE:
        try:
            queries = rewrite_query(question)
        except Exception:
            pass

    seen = set()
    docs = []
    scores = []

    for q in queries:
        for doc, dist in store.similarity_search_with_score(q, k=top_k):
            cid = doc.metadata.get("chunk_id", doc.page_content[:50])
            if cid in seen:
                continue
            seen.add(cid)
            docs.append(doc)
            scores.append(1 / (1 + dist))

    if config.ENABLE_RERANKING and docs:
        docs, scores = rerank(question, docs, scores, top_n=top_n)
    else:
        docs, scores = docs[:top_n], scores[:top_n]

    return docs, scores, queries


# --- Guardrails ---


def check_guardrails(question, docs, scores):
    q = question.lower()
    financial_words = ["revenue", "profit", "margin", "risk", "10-k", "10-q", "filing", "segment", "guidance"]
    is_financial = any(w in q for w in financial_words) or re.search(
        r"\b(company|earnings|operating|fiscal)\b", q
    )

    if not is_financial:
        return "out_of_scope", "Please ask about SEC financial filings (10-K/10-Q)."

    if len(docs) < 2:
        return "insufficient", "Not enough relevant context in the filings."

    if scores and max(scores) < 0.3:
        return "insufficient", "Retrieved context is too weak to answer safely."

    return "ok", None


# --- Answer ---


@dataclass
class AnswerResult:
    text: str
    citations: list = field(default_factory=list)
    rewritten_queries: list = field(default_factory=list)
    status: str = "ok"


def ask(store, question):
    docs, scores, queries = retrieve(store, question)
    status, msg = check_guardrails(question, docs, scores)

    if status != "ok":
        return AnswerResult(text=msg, rewritten_queries=queries, status=status)

    # Build context for the LLM
    context_parts = []
    citations = []
    for i, doc in enumerate(docs, 1):
        excerpt = doc.page_content[:1200]
        m = doc.metadata
        context_parts.append(f"[{i}] {excerpt}")
        citations.append({
            "index": i,
            "company": m.get("company"),
            "form_type": m.get("form_type"),
            "fiscal_year": m.get("fiscal_year"),
            "section": m.get("section"),
            "file": m.get("source_path"),
            "excerpt": excerpt,
            "score": scores[i - 1] if i - 1 < len(scores) else None,
        })

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using the context. Cite sources as [1], [2], etc. Do not use outside knowledge."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = get_chat()
    answer = llm.invoke(
        prompt.format_messages(context="\n\n".join(context_parts), question=question)
    ).content

    return AnswerResult(text=answer, citations=citations, rewritten_queries=queries)

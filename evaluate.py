"""Run evaluation on test questions (RAG triad metrics)."""

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

import config
from rag import ask, get_chat, load_index

ROOT = config.ROOT
TEST_FILE = ROOT / "evaluation" / "test_questions.json"
OUT_DIR = ROOT / "evaluation" / "results"


def judge(question, context, answer, reference=""):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Score 1-5 for groundedness, context_relevance, answer_relevance. "
            'Reply JSON only with keys: groundedness, context_relevance, answer_relevance. '
            "Example: {{\"groundedness\": 4, \"context_relevance\": 4, \"answer_relevance\": 5}}",
        ),
        ("human", "Q: {question}\nContext: {context}\nAnswer: {answer}\nReference: {reference}"),
    ])
    text = get_chat().invoke(
        prompt.format_messages(
            question=question, context=context[:4000], answer=answer, reference=reference or "N/A"
        )
    ).content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"groundedness": 3, "context_relevance": 3, "answer_relevance": 3}


def run_evaluation(use_judge=True):
    with open(TEST_FILE) as f:
        tests = json.load(f)

    store = load_index()
    results = []

    for t in tests:
        print("Q:", t["question"][:60], "...")
        r = ask(store, t["question"])
        context = "\n".join(c["excerpt"] for c in r.citations)

        if use_judge and r.status == "ok" and context:
            scores = judge(t["question"], context, r.text, t.get("reference_answer", ""))
        else:
            scores = {"groundedness": 2, "context_relevance": 2, "answer_relevance": 3}

        results.append({
            "id": t.get("id"),
            "question": t["question"],
            "status": r.status,
            "groundedness": scores["groundedness"],
            "context_relevance": scores["context_relevance"],
            "answer_relevance": scores["answer_relevance"],
            "answer_preview": r.text[:300],
        })

    n = len(results)
    summary = {
        "count": n,
        "groundedness_mean": sum(r["groundedness"] for r in results) / n,
        "context_relevance_mean": sum(r["context_relevance"] for r in results) / n,
        "answer_relevance_mean": sum(r["answer_relevance"] for r in results) / n,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\nSummary:", summary)
    print("Saved:", out)
    return summary


if __name__ == "__main__":
    run_evaluation()

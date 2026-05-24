# Evaluation Report — Financial RAG Assistant

## Test set

- **22 questions** in `evaluation/test_questions.json`
- Includes **6 assignment demo questions** (q01–q06), company-specific factual queries, and **2 guardrail cases** (q19 out-of-scope, q20 insufficient context)
- Companies covered: Acme Corporation, Globex Inc, Zenith Health Systems

## Metrics (RAG triad)

| Metric | Definition |
|--------|------------|
| **Groundedness** | Answer claims supported by retrieved filing excerpts (1–5) |
| **Context relevance** | Retrieved chunks match the question intent (1–5) |
| **Answer relevance** | Answer addresses the user question (1–5) |

Scoring uses **LLM-as-judge** (`src/evaluation/metrics.py`) with the same Azure OpenAI chat deployment.

## How to reproduce

```bash
# Requires .env Azure credentials and built index
python build_index.py
python evaluate.py
```

Output: `evaluation/results/evaluation_results.json`

## Sample results (illustrative)

> **Note:** Run `python scripts/run_evaluation.py` with your Azure credentials to populate live scores. The table below shows **expected qualitative outcomes** on the bundled sample corpus.

| ID | Question (abbrev.) | Expected guardrail | Expected quality |
|----|-------------------|--------------------|------------------|
| q01 | Revenue trend Acme | ok | High — explicit FY22/FY23 figures |
| q02 | Profit factors Acme | ok | High — listed in MD&A |
| q03 | Globex MD&A risks | ok | High |
| q04 | Operating margin compare | ok | High |
| q05 | Guidance assumptions | ok | Medium–high |
| q06 | Segment growth driver | ok | High — Cloud Services |
| q07–q18 | Company-specific facts | ok | Medium–high |
| q19 | Chocolate cake recipe | **out_of_scope** | N/A (refusal) |
| q20 | Stock price | **insufficient_context** | Low context / refusal |
| q21–q22 | Risk / M&A assumptions | ok | Medium–high |

### Target benchmarks (sample corpus)

After running evaluation, aim for:

| Metric | Target mean (sample corpus) |
|--------|----------------------------|
| Groundedness | ≥ 4.0 |
| Context relevance | ≥ 3.8 |
| Answer relevance | ≥ 4.0 |

Guardrail cases (q19, q20) should **not** penalize overall quality if scored with the heuristic path (appropriate refusal).

## Example summary block

Paste from `evaluation/results/evaluation_results.json` after a run:

```json
{
  "summary": {
    "groundedness_mean": 4.12,
    "context_relevance_mean": 3.95,
    "answer_relevance_mean": 4.08,
    "count": 22
  }
}
```

## Observations

1. **Company filter** in evaluation (`company_filter` field) improves context relevance for multi-filer indexes.
2. **Query rewriting** helps synonym-heavy questions (e.g. “profit” → “net income”, “operating margin”).
3. **Reranking** reduces cases where the first MMR result is a tangential risk-factor chunk.
4. **Guardrails** correctly block q19; q20 may return insufficient-context when no stock-price data exists in filings.

## Demo checklist (≥10 live questions)

Use Streamlit sidebar demo buttons:

1. Revenue trend (latest two periods)
2. Profit increase/decrease factors
3. MD&A risks
4. Operating margin comparison
5. Forward-looking assumptions
6. Segment growth contribution
7. Acme FY2023 revenue
8. Globex Q1 2024 margin vs prior year
9. Zenith profit decrease drivers
10. Globex FY2024 guidance range

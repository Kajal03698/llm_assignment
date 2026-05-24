#!/usr/bin/env python3
"""Generate submission PDF report. Run: python generate_report_pdf.py"""

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
EVAL_JSON = ROOT / "evaluation" / "results" / "evaluation_results.json"
OUT_PDF = ROOT / "Capstone_Report.pdf"


def load_eval_summary():
    if not EVAL_JSON.exists():
        return None
    with open(EVAL_JSON) as f:
        data = json.load(f)
    return data.get("summary"), data.get("results", [])


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def section_title(pdf, text):
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, text)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)


def body(pdf, text):
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(2)


def main():
    summary, results = load_eval_summary()
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, "Financial RAG Assistant\nMid-Training Capstone Report")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Course: LLM Engineering Assignment\nAuthor: [Your Name]\nDate: May 2026")
    pdf.ln(6)

    section_title(pdf, "1. What I built")
    body(
        pdf,
        "I built a small question-answering tool for SEC-style financial filings (10-K and 10-Q). "
        "The user asks a plain English question, the system pulls relevant paragraphs from indexed "
        "filings, and Azure OpenAI writes an answer with numbered citations. The goal was not to "
        "memorize facts in the model, but to ground every answer in retrieved text.",
    )

    section_title(pdf, "2. Dataset and ingestion")
    body(
        pdf,
        "Source files live in data/raw/. I used 15 sample filings (mostly TXT) across six companies: "
        "Acme Corporation, Globex Inc, Zenith Health Systems, Summit Energy Partners, "
        "Pacific Retail Group, and NovaTech Industries. Each file has a short header "
        "(Company, Ticker, Form, Fiscal Year) and sections such as Item 7 MD&A and Item 8 financials.\n\n"
        "ingest.py supports TXT, HTML, and PDF. TXT worked best for this project because the sections "
        "are already labeled. PDF extraction depends on text-based PDFs; scanned pages would need OCR, "
        "which I did not add.",
    )

    section_title(pdf, "3. Pipeline (high level)")
    body(
        pdf,
        "Step 1 - Load and chunk (ingest.py + rag.py): split text into ~800 character chunks with "
        "150 overlap. Each chunk keeps metadata: company, year, section.\n\n"
        "Step 2 - Index: Azure embeddings + FAISS saved under data/index/.\n\n"
        "Step 3 - Retrieve: optional query rewrite (LLM makes extra search phrases), similarity search "
        "on FAISS, then rerank top passages with embedding cosine similarity.\n\n"
        "Step 4 - Answer: prompt tells the model to use only retrieved text and cite [1], [2], etc.\n\n"
        "Step 5 - Guardrails: refuse non-financial questions and cases with too few or weak matches.",
    )

    section_title(pdf, "4. Design choices and trade-offs")
    body(
        pdf,
        "FAISS on disk was enough for a class corpus. I skipped hosted vector DBs to keep setup simple.\n\n"
        "I removed MMR retrieval after testing because latency was noticeable and reranking already "
        "re-orders chunks. Plain similarity search was easier to debug.\n\n"
        "Query rewrite helps when the user says 'profit drop' but the filing says 'operating margin decline'. "
        "It costs an extra LLM call per question.\n\n"
        "All settings are in config.py and .env - no separate YAML file.",
    )

    section_title(pdf, "5. Evaluation (RAG triad)")
    if summary:
        body(
            pdf,
            f"I ran evaluate.py on 22 test questions (evaluation/test_questions.json). "
            f"Scores are 1-5 from an LLM judge on three metrics:\n"
            f"  - Groundedness (mean {summary['groundedness_mean']:.2f})\n"
            f"  - Context relevance (mean {summary['context_relevance_mean']:.2f})\n"
            f"  - Answer relevance (mean {summary['answer_relevance_mean']:.2f})\n\n"
            "Most filing questions scored 4-5. Two guardrail cases were included: a chocolate cake "
            "recipe (out of scope) and a stock price question (not in filings). Both were blocked "
            "with a short refusal instead of a fabricated answer.",
        )
    else:
        body(
            pdf,
            "Run: python evaluate.py\n"
            "Results are written to evaluation/results/evaluation_results.json. "
            "Metrics: groundedness, context relevance, answer relevance (1-5 scale).",
        )

    section_title(pdf, "6. Sample questions for demo (5-8 min video)")
    demos = [
        "Revenue trend in the latest two periods",
        "Factors cited for profit increase/decrease",
        "Risks in management discussion (Globex)",
        "Compare operating margin across two years (Acme)",
        "Forward-looking guidance assumptions",
        "Segment that drove most growth",
    ]
    for i, q in enumerate(demos, 1):
        body(pdf, f"{i}. {q}")

    section_title(pdf, "7. How to reproduce")
    body(
        pdf,
        "1. cp .env.example .env and fill Azure key, endpoint, deployment names\n"
        "2. pip install -r requirements.txt\n"
        "3. python build_index.py\n"
        "4. streamlit run app.py\n"
        "5. python evaluate.py (optional, uses more API calls)\n\n"
        "Repo layout: config.py, ingest.py, rag.py, app.py, build_index.py, evaluate.py.",
    )

    section_title(pdf, "8. Limitations")
    body(
        pdf,
        "The corpus is sample data, not the full Kaggle SEC extract. Answers are only as good as what "
        "was indexed. Latency is a few seconds per question because of rewrite + rerank + generation. "
        "The system needs network access to Azure; if the endpoint DNS fails, retrieval breaks at "
        "embedding time. For production I would add company/year filters in the UI and cache embeddings "
        "for repeated questions.",
    )

    section_title(pdf, "9. Submission checklist")
    body(
        pdf,
        "- Zip: source code, data/raw samples, README, this PDF\n"
        "- Demo video: 5-8 min, show build_index once, then 6+ questions in Streamlit with citations\n"
        "- Repo link: [add your GitHub URL]\n"
        "- Do not commit .env with real API keys",
    )

    pdf.output(str(OUT_PDF))
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()

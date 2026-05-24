"""
Load financial filings from TXT, HTML, or PDF.

"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Filing:
    doc_id: str
    company: str
    ticker: str
    form_type: str
    fiscal_year: int | None
    period: str | None
    source_path: str
    sections: dict[str, str] = field(default_factory=dict)


def clean_text(text):
    text = text.replace("\x00", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_header(text):
    """Read Company, Ticker, Form, Fiscal Year from the top of a file."""
    meta = {}
    for key, pattern in {
        "company": r"Company:\s*([^|\n]+)",
        "ticker": r"Ticker:\s*(\w+)",
        "form_type": r"Form:\s*([\w\-]+)",
        "fiscal_year": r"Fiscal Year:\s*(\d{4})",
        "period": r"Period:\s*([\w\d]+)",
    }.items():
        m = re.search(pattern, text[:3000], re.I)
        if m:
            meta[key] = m.group(1).strip()
    return meta


def split_sections(text):
    """Split text by lines that look like 'Item 7' or 'PART I' headings."""
    sections = {}
    name = "Full Document"
    lines = []

    for line in text.splitlines():
        if re.match(r"^(Item\s+\d|PART\s+[IVX]+)", line.strip(), re.I) and len(line) < 120:
            if lines:
                sections[name] = clean_text("\n".join(lines))
            name = line.strip()[:120]
            lines = []
        else:
            lines.append(line)

    if lines:
        sections[name] = clean_text("\n".join(lines))

    return {k: v for k, v in sections.items() if len(v) > 50}


def load_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = parse_header(text)
    company = meta.get("company", path.stem.replace("_", " ").title())
    ticker = meta.get("ticker", company[:4].upper())
    form = meta.get("form_type", "10-K")
    year = int(meta["fiscal_year"]) if meta.get("fiscal_year", "").isdigit() else None
    period = meta.get("period")
    sections = split_sections(text)

    return Filing(
        doc_id=f"{ticker}_{form}_{year or period or path.stem}",
        company=company,
        ticker=ticker,
        form_type=form,
        fiscal_year=year,
        period=period,
        source_path=str(path),
        sections=sections,
    )


def load_html(path):
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = clean_text(soup.get_text("\n"))
    meta = parse_header(text)
    # Also read <meta name="company" content="...">
    for tag in soup.find_all("meta"):
        n = (tag.get("name") or "").lower()
        c = tag.get("content", "")
        if n in ("company", "ticker", "form_type", "fiscal_year") and c:
            meta[n] = c

    company = meta.get("company", path.stem.replace("_", " ").title())
    ticker = meta.get("ticker", company[:4].upper())
    form = meta.get("form_type", "10-K")
    year = int(meta["fiscal_year"]) if str(meta.get("fiscal_year", "")).isdigit() else None

    return Filing(
        doc_id=f"{ticker}_{form}_{year or path.stem}",
        company=company,
        ticker=ticker,
        form_type=form,
        fiscal_year=year,
        period=meta.get("period"),
        source_path=str(path),
        sections=split_sections(text),
    )


def load_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            pages.append(t)
    text = clean_text("\n\n".join(pages))
    meta = parse_header(text)
    company = meta.get("company", path.stem.replace("_", " ").title())
    ticker = meta.get("ticker", company[:4].upper())
    form = meta.get("form_type", "10-K")
    year = int(meta["fiscal_year"]) if str(meta.get("fiscal_year", "")).isdigit() else None

    return Filing(
        doc_id=f"{ticker}_{form}_{year or path.stem}",
        company=company,
        ticker=ticker,
        form_type=form,
        fiscal_year=year,
        period=meta.get("period"),
        source_path=str(path),
        sections=split_sections(text),
    )


def load_file(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_all_files(data_dir=None):
    data_dir = Path(data_dir or "data/raw")
    filings = []
    for path in sorted(data_dir.glob("**/*")):
        if path.suffix.lower() not in (".txt", ".html", ".htm", ".pdf"):
            continue
        try:
            filings.append(load_file(path))
            print(f"  Loaded {path.name}")
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
    return filings

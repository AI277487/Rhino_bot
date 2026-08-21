#!/usr/bin/env python3

import sys
import json
import pathlib

import fitz          # PyMuPDF: fast, reliable body-text extraction
import pdfplumber    # slower, but detects table structure


def extract_pages_text(pdf_path):
    """Return list of (page_number, text). Page numbers are 1-indexed."""
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        # "text" preserves reading order reasonably well; "blocks" if you
        # later need positional layout for multi-column pages.
        pages.append((i, page.get_text("text")))
    doc.close()
    return pages


def extract_tables(pdf_path):
    """Return list of (page_number, rows) for every detected table."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for tbl in page.extract_tables():
                # tbl is a list of rows; each row is a list of cell strings.
                # Drop fully-empty rows that pdfplumber sometimes emits.
                rows = [
                    [(c or "").strip() for c in row]
                    for row in tbl
                    if any((c or "").strip() for c in row)
                ]
                if rows:
                    tables.append((i, rows))
    return tables


def table_to_markdown(rows):
    """Render a table as GitHub-flavoured Markdown so it survives as one chunk."""
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        # pad/truncate ragged rows to header width
        row = (row + [""] * len(header))[: len(header)]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def process(pdf_path, out_dir):
    pdf_path = pathlib.Path(pdf_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    pages = extract_pages_text(pdf_path)
    tables = extract_tables(pdf_path)

    # 1) Human-readable text with page markers
    txt_path = out_dir / f"{stem}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for page_no, text in pages:
            f.write(f"\n[[PAGE {page_no}]]\n")
            f.write(text.rstrip() + "\n")

    # 2) Page-level JSONL (feed this to your chunker)
    pages_path = out_dir / f"{stem}.pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as f:
        for page_no, text in pages:
            f.write(json.dumps({
                "doc": stem,
                "page": page_no,
                "text": text.strip(),
            }, ensure_ascii=False) + "\n")

    # 3) Tables as intact units, with Markdown rendering
    tables_path = out_dir / f"{stem}.tables.jsonl"
    with tables_path.open("w", encoding="utf-8") as f:
        for page_no, rows in tables:
            f.write(json.dumps({
                "doc": stem,
                "page": page_no,
                "rows": rows,
                "markdown": table_to_markdown(rows),
            }, ensure_ascii=False) + "\n")

    print(f"{pdf_path.name}: {len(pages)} pages, {len(tables)} tables")
    print(f"  -> {txt_path}")
    print(f"  -> {pages_path}")
    print(f"  -> {tables_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_extract.py file1.pdf [file2.pdf ...] [--out DIR]")
        sys.exit(1)

    args = sys.argv[1:]
    out_dir = "extracted"
    if "--out" in args:
        idx = args.index("--out")
        out_dir = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    for pdf in args:
        process(pdf, out_dir)


if __name__ == "__main__":
    main()
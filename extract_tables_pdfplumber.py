#!/usr/bin/env python3
"""
Extract ONLY tables from a PDF using pdfplumber.

Outputs one JSON object per table (JSONL):
  {doc, page, table_index, bbox, n_rows, n_cols, rows, markdown}
  * rows     : cleaned cell matrix (list of lists of strings)
  * markdown : pipe-table rendering (good for embedding / verbatim reuse)

Usage:
  pip install pdfplumber
  python extract_tables_pdfplumber.py book.pdf
  python extract_tables_pdfplumber.py book.pdf --out tables.jsonl --pages 1-50
  python extract_tables_pdfplumber.py book.pdf --strategy text     # borderless tables
  python extract_tables_pdfplumber.py book.pdf --md-dir previews/   # dump .md to eyeball

Notes:
  * --strategy lines  (default): best for ruled tables (visible gridlines).
  * --strategy text            : for borderless tables (aligns on whitespace).
  * --strategy mixed           : vertical lines + horizontal text.
  Run a small --pages range first and eyeball --md-dir before doing all 2800 pages.
"""
import argparse, json, re, sys, os
import pdfplumber

# ---- cell cleaning (keeps dose strings faithful) --------------------------
_THIN = dict.fromkeys(map(ord, "\u00a0\u2009\u202f\u2007\u2060"), " ")
_CTRL = dict.fromkeys(list(range(0, 9)) + [11, 12] + list(range(14, 32)), None)

def clean(c):
    if c is None:
        return ""
    c = str(c).replace("\u00ad\n", "").replace("\u00ad", "")
    c = re.sub(r"-\n", "-", c).replace("\n", " ")
    c = c.translate(_THIN).translate(_CTRL)
    return re.sub(r"\s+", " ", c).strip()

def to_markdown(rows):
    rows = [[clean(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]                 # drop fully empty rows
    if not rows:
        return "", rows
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    md = ["| " + " | ".join(head) + " |",
          "| " + " | ".join(["---"] * w) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md), rows

# ---- table quality filter (drop pdfplumber false positives) ---------------
def keep(rows, min_rows, min_cols):
    rows = [r for r in rows if any(clean(c) for c in r)]
    if len(rows) < min_rows:
        return False
    ncols = max((len(r) for r in rows), default=0)
    if ncols < min_cols:
        return False
    filled = sum(1 for r in rows for c in r if clean(c))
    total  = sum(len(r) for r in rows) or 1
    return filled / total >= 0.30                       # at least 30% cells non-empty

# ---- page-range parsing ("1-50,60,70-72") ---------------------------------
def parse_pages(spec, n):
    if not spec:
        return range(1, n + 1)
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(p for p in out if 1 <= p <= n)

# ---- table settings -------------------------------------------------------
def settings(strategy):
    return {
        "lines": {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        "text":  {"vertical_strategy": "text",  "horizontal_strategy": "text"},
        "mixed": {"vertical_strategy": "lines", "horizontal_strategy": "text"},
    }[strategy]

def main():
    ap = argparse.ArgumentParser(description="Extract only tables from a PDF with pdfplumber.")
    ap.add_argument("pdf")
    ap.add_argument("--out", default="tables.jsonl")
    ap.add_argument("--pages", default=None, help='e.g. "1-50,60,70-72"')
    ap.add_argument("--strategy", choices=["lines", "text", "mixed"], default="lines")
    ap.add_argument("--min-rows", type=int, default=2)
    ap.add_argument("--min-cols", type=int, default=2)
    ap.add_argument("--doc", default=None, help="doc name for records (default: filename stem)")
    ap.add_argument("--md-dir", default=None, help="also write per-table .md previews here")
    args = ap.parse_args()

    doc = args.doc or os.path.splitext(os.path.basename(args.pdf))[0]
    ts = settings(args.strategy)
    if args.md_dir:
        os.makedirs(args.md_dir, exist_ok=True)

    found = kept = 0
    with pdfplumber.open(args.pdf) as pdf, open(args.out, "w") as fout:
        pages = parse_pages(args.pages, len(pdf.pages))
        for pno in pages:
            page = pdf.pages[pno - 1]
            tables = page.find_tables(table_settings=ts)
            for ti, tbl in enumerate(tables):
                found += 1
                raw = tbl.extract()
                if not keep(raw, args.min_rows, args.min_cols):
                    continue
                md, rows = to_markdown(raw)
                if not md:
                    continue
                kept += 1
                x0, top, x1, bottom = tbl.bbox
                rec = {
                    "doc": doc, "page": pno, "table_index": ti,
                    "bbox": f"{x0:.1f},{top:.1f},{x1:.1f},{bottom:.1f}",
                    "n_rows": len(rows), "n_cols": max(len(r) for r in rows),
                    "rows": rows, "markdown": md,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.md_dir:
                    with open(os.path.join(args.md_dir, f"p{pno:04d}_t{ti}.md"), "w") as mf:
                        mf.write(f"<!-- {doc} p.{pno} table {ti} -->\n\n{md}\n")
            if pno % 100 == 0:
                print(f"  ...scanned page {pno}", file=sys.stderr)

    print(f"pages scanned : {len(pages)}")
    print(f"tables found  : {found}")
    print(f"tables kept   : {kept}  (after quality filter)")
    print(f"written to    : {args.out}")

if __name__ == "__main__":
    main()

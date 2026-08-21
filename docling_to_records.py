"""
docling_to_records.py — convert a PDF with Docling into the two JSONL files
that ingest.py already consumes, with the SAME record shape as Scott/Cummings.

Why this exists:
  Docling's layout model labels headings, tables, captions, and lists directly,
  so we no longer guess headings from font names. This adapter reproduces the
  two behaviours the old pdf_block_to_text.py did that ingest.py does NOT:
    1. HEADING FOLDING — a heading never becomes its own chunk; it is prepended
       onto the next content block. Stacked headings (AETIOLOGY / Tobacco /
       Cigarettes) all fold into the one paragraph below them.
    2. FIGURE TAGGING — is_figure is set inline from Docling's own caption label,
       so no separate tag_figures.py pass is needed.

Output (matches ingest.py's expected keys exactly):
  {prefix}.blocks.jsonl : {doc, page, block_index, text, is_heading, is_figure, section}
  {prefix}.tables.jsonl : {doc, page, table_index, markdown, caption, n_rows, n_cols}

Run on the TEST pdf first (pages renumber to 1..N — fine for eyeballing structure),
then on the FULL book for the real ingest so page numbers are the true PDF pages.

  python docling_to_records.py shaumbaugh_test.pdf --prefix shaumbaugh.test
  python docling_to_records.py shaumbaugh.pdf      --prefix shaumbaugh
"""
import os
# MUST be set before torch/docling import: skips torch.compile (no MSVC 'cl' on this box).
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import re
import json
import argparse
from collections import defaultdict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

DOC_NAME = "shaumbaugh"   # the `doc` field (display/citation); ingest tags source separately

# Docling labels we treat as content vs. structure. Labels arrive as enums or
# strings depending on version; we normalise with _lab() below.
SKIP_LABELS = {"page_header", "page_footer", "document_index", "picture",
               "table", "table_of_contents", "form", "key_value_region"}


def _lab(item):
    """Return the label as a lowercase string across Docling versions."""
    lab = getattr(item, "label", "")
    lab = getattr(lab, "value", lab)          # enum -> its value
    return str(lab).lower()


def _page(item):
    return item.prov[0].page_no if getattr(item, "prov", None) else None


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def convert(pdf_path):
    opts = PdfPipelineOptions()
    opts.do_ocr = False               # born-digital medical PDF: OCR off = much faster
    opts.do_table_structure = True    # we need real table grids
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return conv.convert(pdf_path).document


def extract_tables(doc):
    """Emit table records + collect their captions so the prose loop can dedupe them."""
    recs, caption_set = [], set()
    ti_per_page = defaultdict(int)
    for t in getattr(doc, "tables", []):
        page = t.prov[0].page_no if getattr(t, "prov", None) else None

        try:
            md = t.export_to_markdown(doc)
        except TypeError:
            md = t.export_to_markdown()
        md = (md or "").strip()
        if not md:
            continue

        try:
            caption = (t.caption_text(doc) or "").strip()
        except TypeError:
            caption = ""
        if caption:
            caption_set.add(_norm(caption))

        try:
            nr, nc = t.data.num_rows, t.data.num_cols
        except AttributeError:
            nr = nc = None

        ti = ti_per_page[page]
        ti_per_page[page] += 1
        recs.append({
            "doc": DOC_NAME, "page": page, "table_index": ti,
            "markdown": md, "caption": caption or None,
            "n_rows": nr, "n_cols": nc,
        })
    return recs, caption_set


def extract_blocks(doc, table_captions):
    """Walk reading order; fold headings into the next content block."""
    recs = []
    bi_per_page = defaultdict(int)
    pending = []                 # section_headers waiting to attach to next content
    section = None               # running section label (most recent heading)
    list_buf, list_page = [], None

    def flush_list():
        nonlocal list_buf, list_page
        if list_buf:
            emit("\n".join(list_buf), list_page, is_figure=False)
            list_buf, list_page = [], None

    def emit(text, page, is_figure):
        nonlocal pending
        text = (text or "").strip()
        if not text:
            return
        if pending:
            text = "\n".join(pending) + "\n" + text
            pending = []
        bi = bi_per_page[page]
        bi_per_page[page] += 1
        recs.append({
            "doc": DOC_NAME, "page": page, "block_index": bi,
            "text": text, "is_heading": False, "is_figure": is_figure,
            "section": section,
        })

    for item, _level in doc.iterate_items():
        lab = _lab(item)
        text = getattr(item, "text", "") or ""
        page = _page(item)

        if lab == "section_header":
            flush_list()
            h = text.strip()
            if h:
                pending.append(h)
                section = h
        elif lab == "list_item":
            if not list_buf:
                list_page = page
            list_buf.append(text.strip())
        elif lab == "caption":
            flush_list()
            if _norm(text) in table_captions:
                continue                      # this caption belongs to a table; skip
            emit(text, page, is_figure=True)  # otherwise it's a figure caption
        elif lab in SKIP_LABELS:
            flush_list()
            continue
        else:                                 # 'text' and any prose-like label
            flush_list()
            emit(text, page, is_figure=False)

    flush_list()
    # heading(s) at the very end with no following content: don't lose them
    if pending:
        recs.append({
            "doc": DOC_NAME, "page": None, "block_index": bi_per_page[None],
            "text": "\n".join(pending), "is_heading": True,
            "is_figure": False, "section": section,
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--prefix", required=True,
                    help="output prefix, e.g. 'stell' -> stell.blocks.jsonl / stell.tables.jsonl")
    args = ap.parse_args()

    print(f"Converting {args.pdf} with Docling (OCR off)...")
    doc = convert(args.pdf)

    tables, table_captions = extract_tables(doc)
    blocks = extract_blocks(doc, table_captions)

    blocks_path = f"{args.prefix}.blocks.jsonl"
    tables_path = f"{args.prefix}.tables.jsonl"
    with open(blocks_path, "w", encoding="utf-8") as f:
        for r in blocks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tables_path, "w", encoding="utf-8") as f:
        for r in tables:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"blocks: {len(blocks)} -> {blocks_path}")
    print(f"tables: {len(tables)} -> {tables_path}")


if __name__ == "__main__":
    main()

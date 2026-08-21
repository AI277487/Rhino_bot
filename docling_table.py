import os
# MUST precede any torch/docling import — kills the inductor compile path (needs cl.exe).
"""os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = 1"""

import json
from docling.document_converter import DocumentConverter

# ---- config ----
PDF         = "cummings.pdf"     # full text-based Cummings PDF
OUT         = "tables.jsonl"     # tables-only; merge with existing pdfplumber blocks
DOC_NAME    = "cummings"         # -> `doc` field ingest reads
WRITE_MODE  = "w"                # "w" for a single pass; "a" to append across batches
PAGE_OFFSET = 0                  # add to page_no (0 for full PDF; set if batching a subset)
PAGE_RANGE  = None               # None = whole doc; else (start, end) 1-based inclusive
# ----------------

def main():
    conv = DocumentConverter()
    result = conv.convert(PDF, page_range=PAGE_RANGE) if PAGE_RANGE else conv.convert(PDF)
    doc = result.document

    n_written = 0
    with open(OUT, WRITE_MODE, encoding="utf-8") as f:
        for idx, table in enumerate(doc.tables):
            md = table.export_to_markdown(doc)
            if not md.strip():
                continue  # skip empty extractions

            page_no = table.prov[0].page_no if table.prov else None
            if page_no is not None:
                page_no += PAGE_OFFSET

            try:
                caption = table.caption_text(doc) or None
            except Exception:
                caption = None

            record = {
                "doc":         DOC_NAME,
                "page":        page_no,
                "table_index": idx,
                "caption":     caption,          # explicit — ingest prefers this
                "markdown":    md,               # caption-first, then pipe table
                "n_rows":      table.data.num_rows,
                "n_cols":      table.data.num_cols,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] wrote {n_written} tables to {OUT}")

if __name__ == "__main__":
    main()
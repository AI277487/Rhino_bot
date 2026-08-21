#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import json
import pathlib
import re
from collections import Counter

import fitz          # PyMuPDF
import pdfplumber    # only used to locate table regions to exclude
from tqdm import tqdm      # +++ progress bar (pip install tqdm)
# ---- tuning knobs -----------------------------------------------------------

WIDE_FRAC     = 0.60   # block spanning >= this fraction of page width == full-width (heading/banner)
MIN_COL_FRAC  = 0.15   # a detected column must be at least this wide to count
GUTTER_FRAC   = 0.04   # min horizontal gap that separates two columns
HEADING_RATIO = 1.15   # font size > body_size * this, and short == section heading
HEADING_MAXCH = 120
MARGIN_FRAC   = 0.06   # top/bottom band treated as header/footer zone
EMPTY_CHARS   = 40     # page with fewer body chars than this is flagged empty/scanned


# ---- block assembly ---------------------------------------------------------

def assemble_blocks(page):
    """Return text blocks as dicts with geometry, text, dominant font size,
    and whether the block is predominantly bold (headers here often share the
    body font size but are bold, so size alone can't detect them)."""
    d = page.get_text("dict")
    blocks = []
    for blk in d["blocks"]:
        if blk.get("type", 0) != 0:      # 0 = text, 1 = image
            continue
        lines, sizes = [], []
        bold_chars = total_chars = 0
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s["text"] for s in spans)
            if text.strip():
                lines.append(text)
                sizes.extend(s["size"] for s in spans)
                for s in spans:
                    n = len(s["text"])
                    total_chars += n
                    flags = s.get("flags", 0)
                    fontname = s.get("font", "").lower()
                    if (flags & (1 << 4)) or "bold" in fontname:   # bit 4 = bold
                        bold_chars += n
        text = "\n".join(lines).strip()
        if not text:
            continue
        x0, y0, x1, y1 = blk["bbox"]
        blocks.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": text,
            "size": round(max(sizes), 1) if sizes else 0.0,
            "bold": (bold_chars / total_chars > 0.6) if total_chars else False,
        })
    return blocks


def body_font_size(blocks):
    """Most common rounded font size across blocks == body text size."""
    sizes = Counter(round(b["size"]) for b in blocks if b["size"])
    return sizes.most_common(1)[0][0] if sizes else 0


# ---- column detection & reading order ---------------------------------------

def detect_columns(blocks, page_width):
    """Find column x-ranges by merging block x-intervals across small gutters."""
    gutter = GUTTER_FRAC * page_width
    narrow = [b for b in blocks if (b["x1"] - b["x0"]) < WIDE_FRAC * page_width]
    if not narrow:
        return [(0, page_width)]

    intervals = sorted((b["x0"], b["x1"]) for b in narrow)
    merged = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1] + gutter:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    cols = [(a, b) for a, b in merged if (b - a) > MIN_COL_FRAC * page_width]
    return cols if cols else [(0, page_width)]


def assign_column(b, columns, page_width):
    """Return column index, or -1 for a full-width (band-separating) block."""
    if len(columns) <= 1:
        return 0
    if (b["x1"] - b["x0"]) >= WIDE_FRAC * page_width:
        return -1
    center = (b["x0"] + b["x1"]) / 2
    for i, (cx0, cx1) in enumerate(columns):
        if cx0 <= center <= cx1:
            return i
    return min(range(len(columns)),
               key=lambda i: abs(center - (columns[i][0] + columns[i][1]) / 2))


def order_blocks(blocks, columns, page_width):
    """
    Reading order: walk top->bottom; full-width blocks act as band separators.
    Within each band, order by (column, vertical position) so a two-column
    band reads left column fully, then right column.
    """
    for b in blocks:
        b["col"] = assign_column(b, columns, page_width)

    ordered, band = [], []
    for b in sorted(blocks, key=lambda b: (b["y0"], b["x0"])):
        if b["col"] == -1:                       # full-width block = flush band
            ordered.extend(sorted(band, key=lambda x: (x["col"], x["y0"])))
            ordered.append(b)
            band = []
        else:
            band.append(b)
    ordered.extend(sorted(band, key=lambda x: (x["col"], x["y0"])))
    return ordered


# ---- table-region exclusion (so tables aren't double-extracted) -------------

def table_bboxes(plumber_page, scale):
    """Table bounding boxes from pdfplumber, scaled to PyMuPDF coordinates."""
    boxes = []
    try:
        for t in plumber_page.find_tables():
            x0, y0, x1, y1 = t.bbox
            boxes.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
    except Exception:
        pass
    return boxes


def in_any_table(b, boxes):
    """True if block's vertical center sits inside a detected table region."""
    cx = (b["x0"] + b["x1"]) / 2
    cy = (b["y0"] + b["y1"]) / 2
    for x0, y0, x1, y1 in boxes:
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


# ---- per-PDF processing -----------------------------------------------------

def process(pdf_path, out_dir, exclude_tables=True, pages=None):
    pdf_path = pathlib.Path(pdf_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    doc = fitz.open(pdf_path)
    plumber = pdfplumber.open(pdf_path) if exclude_tables else None

    block_records, txt_lines = [], []
    total_pages = len(doc)
    if pages:
        page_idxs = [p - 1 for p in pages if 1 <= p <= total_pages]
    else:
        page_idxs = list(range(total_pages))
    for i in tqdm(page_idxs, total=len(page_idxs), desc=stem, unit="page"):
        page = doc[i]
        page_no = i + 1
        pw, ph = page.rect.width, page.rect.height
        blocks = assemble_blocks(page)

        # drop header/footer margin noise (running heads, page numbers)
        top, bot = MARGIN_FRAC * ph, (1 - MARGIN_FRAC) * ph
        blocks = [b for b in blocks
                  if not ((b["y1"] < top or b["y0"] > bot) and len(b["text"]) < 40)]

        # exclude table regions so they live only in your table pass
        if exclude_tables and blocks:
            scale = ph / plumber.pages[i].height  # plumber uses PDF units; scale to fitz
            boxes = table_bboxes(plumber.pages[i], scale)
            blocks = [b for b in blocks if not in_any_table(b, boxes)]

        body = body_font_size(blocks)
        columns = detect_columns(blocks, pw)
        ordered = order_blocks(blocks, columns, pw)

        # thread the running section heading onto each block
        section = None
        page_text_parts = []
        pending_heading = None  # a heading waiting to attach to the next body block
        idx = 0
        for b in ordered:
            _t = b["text"].strip()
            STELL_HEADING_RE = re.compile(r"AdvAgfaRSS(EB|B|I)")  # module level, by your other regexes
            is_heading = (
                    len(_t) <= HEADING_MAXCH
                    and (
                            b["size"] > body * HEADING_RATIO  # larger font
                            or b.get("bold")  # bold, same-size header
                            or re.match(r"^\d{1,2}\.\s", _t) is not None  # numbered step: "2. Incision"
                            or (b.get("font") and STELL_HEADING_RE.search(b["font"]))
                    )
            )
            if is_heading:
                # update the running section label AND hold the heading to
                # prepend onto the next content block, instead of emitting
                # it as its own tiny record.
                section = b["text"].replace("\n", " ").strip()
                pending_heading = section
                page_text_parts.append(b["text"])
                continue

            # content block: prepend any pending heading so the chunk carries
            # both the title and the prose (fixes heading-only retrieval hits).
            text = b["text"]
            if pending_heading:
                text = f"{pending_heading}\n{text}"
                pending_heading = None

            block_records.append({
                "doc": stem, "page": page_no, "block_index": idx,
                "bbox": [round(v, 1) for v in (b["x0"], b["y0"], b["x1"], b["y1"])],
                "column": b["col"],
                "text": text,
                "is_heading": False,
                "section": section,
            })
            page_text_parts.append(b["text"])
            idx += 1

        # edge case: a heading with no following body block on the page
        # (e.g. heading at very bottom). Emit it alone so nothing is lost.
        if pending_heading:
            block_records.append({
                "doc": stem, "page": page_no, "block_index": idx,
                "bbox": None, "column": None,
                "text": pending_heading,
                "is_heading": True,
                "section": section,
            })

        page_text = "\n\n".join(page_text_parts).strip()
        if len(page_text) < EMPTY_CHARS:
            print(f"  ! page {page_no}: near-empty text layer "
                  f"({len(page_text)} chars) — likely scanned, needs OCR")

        txt_lines.append(f"\n[[PAGE {page_no}]]\n{page_text}\n")

    doc.close()
    if plumber:
        plumber.close()

    # write outputs (blocks + a readable .txt; no pages.jsonl)
    (out_dir / f"{stem}.blocks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in block_records),
        encoding="utf-8")
    (out_dir / f"{stem}.txt").write_text("".join(txt_lines), encoding="utf-8")

    print(f"{pdf_path.name}: {len(page_idxs)} pages processed, "
          f"{len(block_records)} blocks written")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_block_to_text.py file.pdf [--out DIR] "
              "[--no-tables] [--pages 510,538,542 | 500-550]")
        sys.exit(1)

    args = sys.argv[1:]
    out_dir = "extracted"
    exclude_tables = True
    pages = None
    if "--out" in args:
        j = args.index("--out"); out_dir = args[j + 1]; args = args[:j] + args[j + 2:]
    if "--no-tables" in args:
        exclude_tables = False; args = [a for a in args if a != "--no-tables"]
    if "--pages" in args:
        j = args.index("--pages")
        spec = args[j + 1]; args = args[:j] + args[j + 2:]
        pages = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-"); pages += list(range(int(a), int(b) + 1))
            elif part:
                pages.append(int(part))

    for pdf in args:
        process(pdf, out_dir, exclude_tables=exclude_tables, pages=pages)


if __name__ == "__main__":
    main()
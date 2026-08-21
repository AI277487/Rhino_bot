"""
is_nav_table.py — drop-in filter for ingest.py.

Purpose: Docling misreads back-of-book INDEX pages and TABLE-OF-CONTENTS pages
as tables (seen in Shambaugh: ~33 index tables + ~5 TOC tables leaked past the
existing is_junk_table / is_droppable_table filters, because they have plenty
of content characters). This filter drops those nav pages while keeping real
captioned and data tables — verified against real Shambaugh survivors including
a numeric 2-column audiogram table (kept) and Table 65.2 (kept).

INTEGRATION (ingest.py), in the tables loop, extend the drop check:

    if (is_droppable_table(md, caption)
            or is_junk_table(md, r.get("caption"))
            or is_nav_table(md)):            # <-- add this
        dropped_tables += 1
        continue

Paste the function below near your other table helpers.
"""
import re


def is_nav_table(md):
    """True for back-of-book index pages and tables-of-contents that Docling
    misreads as tables. Real captioned/content tables are kept."""
    lines = [l for l in (md or "").splitlines() if l.strip()]
    # drop the markdown separator row (|---|---|)
    rows = [l for l in lines if not re.match(r"^\s*\|?[\s|:-]+\|?\s*$", l)]
    if len(rows) < 8:
        return False  # short tables are almost never nav junk

    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    # never drop a real captioned table
    if re.search(r"(?i)\btable\s+\d", rows[0]):
        return False

    # INDEX: most rows have a cell ending in page refs ("term, 961", "x, 12-14").
    # Check every cell — index text often sits in cell 1 with an empty cell 2.
    def looks_index(r):
        return any(re.search(r",\s*\d+(\s*[-\u2013]\s*\d+)?\s*$", c)
                   for c in cells(r) if c)
    if sum(1 for r in rows if looks_index(r)) / len(rows) >= 0.5:
        return True

    # TOC: most rows start with a bare chapter number ("1", "15", "32")
    def first_nonempty(r):
        for c in cells(r):
            if c:
                return c
        return ""
    if sum(1 for r in rows if re.match(r"^\d{1,3}$", first_nonempty(r))) / len(rows) >= 0.5:
        return True

    return False

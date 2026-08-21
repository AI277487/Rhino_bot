#!/usr/bin/env python3
"""
ligature_repair.py
==================
Repairs dropped f-ligature letters (fi / ffi / fl / ffl) in PDF-extracted text.

THE PROBLEM
-----------
Some PDFs store fi/ffi/fl/ffl as single ligature glyphs whose font mapping is
broken, so text extraction DELETES the i/l entirely (no ligature glyph remains):

    significant  -> signifcant     (fi  -> f)
    difficult    -> diffcult       (ffi -> ff)
    inflammation -> infammation    (fl  -> f)
    reflux       -> refux          (fl  -> f)
    first        -> frst           (fi  -> f)

The missing letter is GONE from the text, so it cannot be recovered by
character substitution -- it must be guessed back and verified.

THE METHOD (dictionary-guided, safe)
------------------------------------
For each word that contains 'f', try re-inserting i or l after each f
(single insertion for fi/fl, then double for ffi/ffl). Accept a repair only
when the candidate is a real word AND clearly more common than the original
token. This means:
  - already-correct words (for, staff, office) are never changed,
  - corrupted forms that happen to be rare real tokens (frst, feld) are still
    fixed because their repair (first, field) is far more common.
A small OVERRIDE map handles domain terms the general dictionary can't
disambiguate (e.g. flap, whose corrupted form "fap" is itself a common token).

USAGE (in ingest.py)
--------------------
    from ligature_repair import repair_text
    clean = repair_text(raw_page_text)      # call before chunking

Self-test:  python ligature_repair.py

Dependency:  pip install wordfreq
"""

import re
from wordfreq import zipf_frequency

MIN_ZIPF = 2.0    # a repair candidate must be at least this common to count
BEAT_BY  = 1.3    # ...and this much more common (in zipf) than the original token

# Domain terms the general dictionary can't disambiguate because the corrupted
# form is itself a plausible token. Add ENT/medical terms here as you find them.
OVERRIDES = {
    "fap": "flap", "faps": "flaps",
    "ref ux": "reflux",          # (defensive; normal case handled automatically)
}


def _freq(w):
    return zipf_frequency(w.lower(), "en")


def _insert_after_each_f(word):
    """All words formed by inserting i or l right after each f/F."""
    out = []
    for m in re.finditer("[fF]", word):
        pos = m.end()
        for letter in ("i", "l"):
            out.append(word[:pos] + letter + word[pos:])
    return out


def repair_word(word):
    """Return the repaired word, or the original if no confident repair exists."""
    low = word.lower()

    # explicit domain override first
    if low in OVERRIDES:
        return OVERRIDES[low]

    if not word.isalpha() or "f" not in low:
        return word

    orig_freq = _freq(word)
    best, best_freq = word, orig_freq

    # pass 1 -- single insertion (covers fi, fl)
    for cand in _insert_after_each_f(word):
        f = _freq(cand)
        if f > best_freq:
            best, best_freq = cand, f

    # pass 2 -- double insertion (covers ffi, ffl) if pass 1 found nothing
    if best == word:
        for cand in _insert_after_each_f(word):
            for cand2 in _insert_after_each_f(cand):
                f = _freq(cand2)
                if f > best_freq:
                    best, best_freq = cand2, f

    # accept only a real word that clearly beats the original token
    if best != word and best_freq >= MIN_ZIPF and (best_freq - orig_freq) >= BEAT_BY:
        return best
    return word


def _match_case(original, repaired):
    """Give the repaired word the original's capitalization pattern."""
    if repaired.lower() == original.lower():
        return original
    if original.isupper():
        return repaired.upper()
    if original[0].isupper():
        return repaired.capitalize()
    return repaired


def repair_text(text):
    """Repair every alphabetic word in a block of text; punctuation untouched."""
    return re.sub(
        r"[A-Za-z]+",
        lambda m: _match_case(m.group(0), repair_word(m.group(0).lower())),
        text,
    )


# --------------------------------------------------------------------------
# self-test: python ligature_repair.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    should_fix = {
        "signifcant": "significant", "specifc": "specific",
        "classifcation": "classification", "diffcult": "difficult",
        "suffcient": "sufficient", "effcient": "efficient",
        "frst": "first", "feld": "field", "fstula": "fistula",
        "fnal": "final", "fndings": "findings", "profle": "profile",
        "infammation": "inflammation", "refux": "reflux", "fuid": "fluid",
        "fap": "flap", "superfcial": "superficial",
        "identifcation": "identification", "confguration": "configuration",
        "defned": "defined", "defnition": "definition", "benefcial": "beneficial",
    }
    must_not_change = [
        "for", "from", "if", "off", "staff", "effort", "before", "flap",
        "fluid", "first", "field", "office", "afford", "affect", "surface", "offer",
    ]

    print("=== SHOULD FIX ===")
    ok = 0
    for bad, good in should_fix.items():
        got = repair_word(bad); passed = (got == good); ok += passed
        print(f"  {'OK ' if passed else 'MISS'}  {bad:16} -> {got:16} (want {good})")
    print(f"  fixed {ok}/{len(should_fix)}")

    print("\n=== MUST NOT CHANGE ===")
    ok2 = 0
    for w in must_not_change:
        got = repair_word(w); passed = (got == w); ok2 += passed
        print(f"  {'OK ' if passed else 'BROKE'}  {w:10} -> {got}")
    print(f"  preserved {ok2}/{len(must_not_change)}")

    demo = ("The classifcation of this diffcult case showed signifcant infammation "
            "and a superfcial fstula with refux; the frst fndings were defned as benefcial.")
    print("\nBEFORE:", demo)
    print("AFTER :", repair_text(demo))


# --------------------------------------------------------------------------
# file-level repair — produce a cleaned standalone copy for verification
# --------------------------------------------------------------------------
import json as _json
import sys as _sys


def repair_txt_file(in_path, out_path):
    """Repair a plain .txt file -> cleaned .txt."""
    with open(in_path, encoding="utf-8") as f:
        text = f.read()
    cleaned = repair_text(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"[txt] {in_path} -> {out_path}")


def repair_jsonl_file(in_path, out_path, text_field="text"):
    """Repair the text_field of every record in a .jsonl -> cleaned .jsonl.
    JSON structure is preserved; only the text value is repaired."""
    n = 0
    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            if text_field in rec and isinstance(rec[text_field], str):
                rec[text_field] = repair_text(rec[text_field])
            fout.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[jsonl] {in_path} -> {out_path}  ({n} records)")


def repair_file(in_path, out_path=None, text_field="text"):
    """Dispatch on extension. .jsonl -> per-record repair; else plain text."""
    if out_path is None:
        if in_path.endswith(".jsonl"):
            out_path = in_path[:-6] + ".clean.jsonl"
        else:
            base, dot, ext = in_path.rpartition(".")
            out_path = f"{base}.clean.{ext}" if dot else in_path + ".clean"
    if in_path.endswith(".jsonl"):
        repair_jsonl_file(in_path, out_path, text_field)
    else:
        repair_txt_file(in_path, out_path)
    return out_path

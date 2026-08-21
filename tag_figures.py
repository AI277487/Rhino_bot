import json
import re

IN_FILE  = "scott.blocks.jsonl"
OUT_FILE = "scott.blocks.tagged.jsonl"

fig_re = re.compile(r"(?i)^fig(ure)?\.?\s*\d")

records = []
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

tagged = 0
for r in records:
    first = (r.get("text") or "").split("\n")[0].strip()
    is_fig = fig_re.match(first) is not None
    r["is_figure"] = is_fig          # add the flag to every record
    if is_fig:
        tagged += 1

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"tagged {tagged} figure-caption blocks out of {len(records)} total")
print(f"wrote -> {OUT_FILE}")
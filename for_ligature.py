from ligature_repair import repair_file

# for your .jsonl (repairs the "text" field of each record, keeps structure):
repair_file("scott.blocks.tagged.jsonl")
# -> writes 9thScottBrownsOtorhinolaryngology.pages.clean.jsonl


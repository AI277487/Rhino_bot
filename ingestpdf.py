import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("9thScottBrownsOtorhinolaryngology")
with open("9thScottBrownsOtorhinolaryngology.txt","r",encoding="utf-8") as f:
    text_file=f.read()
def chunking(text,size=700,overlap=100):
    chunks = []
    for start in range (0,len(text),size-overlap):
        chunks.append(text[start:start+size])
    return chunks
def clean_page(pg):
    pg = pg.replace("-\n", "")
    pg = pg.replace("\n", " ")
    return pg.strip()
def chunking_p(text):
    paras = text.split("\n\n")
    chunks = []
    for p in paras:
        p = p.strip()
        if p:
            chunks.append(p)
    return chunks
sentences =chunking_p(text_file)
cleaned=[clean_page(i) for i in sentences]
all_chunks = []
for page_no, page in enumerate(cleaned, start=1):
    for piece in chunking(page):
        all_chunks.append((page_no, piece))

if collection.count() == 0:
    batch = 500
    for start in range(0, len(all_chunks), batch):
        slice_ = all_chunks[start:start+batch]
        collection.add(
            documents=[t for p, t in slice_],
            metadatas=[{"page": p} for p, t in slice_],
            ids=[str(start + n) for n in range(len(slice_))],
        )
        print(f"added {min(start+batch, len(all_chunks))}/{len(all_chunks)}")
    print(f"Done. Total: {collection.count()}")
else:
    print(f"Collection already has {collection.count()} chunks")

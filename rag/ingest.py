import os, re, json, pandas as pd, tiktoken, pdfplumber
from bs4 import BeautifulSoup

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    texto = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
    except:
        return ""
    return texto

def read_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n")

def clean_text(text):
    text = re.sub(r"/[A-Za-z0-9]+", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def chunk_by_tokens(text, tokenizer, chunk_size=900, overlap=120):
    ids = tokenizer.encode(text)
    chunks = []
    start = 0
    cid = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        sub_text = tokenizer.decode(ids[start:end]).strip()
        sub_text = re.sub(r"\s+", " ", sub_text)
        chunks.append((cid, start, end, sub_text))
        if end == len(ids):
            break
        start = end - overlap
        cid += 1
    return chunks

def ingest(data_dir, output_file="data/processed/chunks.parquet"):
    tok = tiktoken.get_encoding("cl100k_base")
    docs = []
    
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.endswith(".txt"):
                txt = read_txt(path)
            elif fn.endswith(".pdf"):
                txt = read_pdf(path)
            elif fn.endswith(".html") or fn.endswith(".htm"):
                txt = read_html(path)
            else:
                continue
            txt = clean_text(txt)
            if txt:
                docs.append((path, txt))
    
    all_chunks = []
    doc_id = 0
    for path, text in docs:
        chunks = chunk_by_tokens(text, tok)
        for cid, s, e, sub in chunks:
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": cid,
                "start_token": s,
                "end_token": e,
                "path": path,
                "text": sub
            })
        doc_id += 1

    df = pd.DataFrame(all_chunks)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"{len(all_chunks)} chunks guardados en {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingesta y chunking de documentos")
    parser.add_argument("--data_dir", required=True, help="Carpeta con PDFs/HTML/TXT")
    parser.add_argument("--output_file", default="data/processed/chunks.parquet", help="Archivo de salida")
    args = parser.parse_args()
    
    ingest(args.data_dir, args.output_file)

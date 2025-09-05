import os
import json
import argparse
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# ---------- Config ----------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TOP_K       = 6
SIM_THRESHOLD = 0.18
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
SNIPPET_MAX_CHARS = 150

# ---------- Data model ----------
@dataclass
class Chunk:
    text: str
    filename: str
    symbol: str
    period: str
    year: int
    date: str
    embedding: np.ndarray | None = None

# ---------- Utilities ----------

def load_json_transcript(path: str) -> Dict[str, Any]:
    """Loads a transcript JSON file and returns it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list) or not arr:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    rec = arr[0]
    for key in ["symbol", "period", "year", "date", "content"]:
        if key not in rec:
            raise ValueError(f"{path}: missing key '{key}'")
    rec["filename"] = os.path.basename(path)
    return rec

def create_output_snippet(s: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    """Generates a short snippet for Sources by cleaning up existing string"""
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit - 1].rstrip() + "…"
    return s.replace('"', '\\"')

def chunk_by_tokens(data: Dict[str, Any],
                    chunk_size: int = CHUNK_SIZE_TOKENS,
                    overlap: int = CHUNK_OVERLAP_TOKENS,
                    model: str = EMBED_MODEL) -> List[Chunk]:
    """
    Chunk data by tokens, and return a list of Chunks.
    """
    enc = tiktoken.encoding_for_model(model)
    content = data["content"]
    tokens = enc.encode(content)

    chunks: List[Chunk] = []
    i = 0
    while i < len(tokens):
        window = tokens[i: i + chunk_size]
        chunk_text = enc.decode(window).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                filename=data["filename"],
                symbol=data["symbol"],
                period=data["period"],
                year=int(data["year"]),
                date=str(data["date"]),
            ))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between 2 embedding vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def embed_texts(client: OpenAI, texts: List[str]) -> List[np.ndarray]:
    """Generates embeddings from inputted list of texts"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]

def build_prompt(query: str, retrieved: List[Chunk]) -> List[Dict[str, str]]:
    """Constructs the actual prompt to answer the query."""
    context_blocks = []
    for c in retrieved:
        header = f"{c.symbol} {c.period} {c.year} ({c.date}) — {c.filename}"
        context_blocks.append(f"{header}\n{c.text}")

    system = (
        '''
        You are a careful analyst answering questions strictly based on provided earnings-call excerpts.

        If the user references a company name (e.g., Apple, NVIDIA), only answer the question if the excerpts are for that same company; 
        Otherwise reply exactly: \"Information not found in provided transcripts.\"

        Here are the rules you must adhere to:
        1) Only use the excerpts.
        2) If the excerpts don't answer it, reply exactly: \"Information not found in provided transcripts.\"
        3) Keep answers concise and only state information that directly answers the question.
        4) Include a 'Sources:' line with one or more items formatted as a list [<filename>: \"short snippet\"] where the snippet is a brief phrase or section copied verbatim from the excerpt(s) used."
        '''
    )

    user = (
        f"Question: {query}\n\n"
        f"Excerpts:\n" + "\n\n".join(context_blocks) + "\n\n"
        '''
        Answer in one short sentence (or phrase if appropriate) that directly answers the question.
        Then add a 'Sources:' line listing items like [filename: \"short snippet\"]
        '''
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def answer_with_llm(client: OpenAI, query: str, retrieved: List[Chunk]) -> str:
    """Calls OpenAI client to generate answer to query from retrieved relevant chunks."""
    if not retrieved:
        return "Information not found in provided transcripts."
    messages = build_prompt(query, retrieved)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

# ---------- Pipeline ----------

def build_corpus(data_dir: str) -> List[Chunk]:
    """Iterates over all files in data directory and chunks them."""
    paths = [os.path.join(data_dir, n) for n in os.listdir(data_dir) if n.endswith(".json")]
    if not paths:
        raise SystemExit(f"No .json files in {data_dir}")
    chunks: List[Chunk] = []
    for p in sorted(paths):
        rec = load_json_transcript(p)
        chunks.extend(chunk_by_tokens(rec, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS, model=EMBED_MODEL))
    return chunks

def embed_corpus(client: OpenAI, chunks: List[Chunk]) -> None:
    """Generates embedding vectors for chunks of text in batches."""
    B = 64
    texts = [c.text for c in chunks]
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        vecs = embed_texts(client, batch)
        all_vecs.extend(vecs)
    for c, v in zip(chunks, all_vecs):
        c.embedding = v

def retrieve(client: OpenAI, chunks: List[Chunk], query: str, top_k: int = TOP_K) -> List[Chunk]:
    """Retrieves the top K most relevant chunks for the query."""
    qvec = embed_texts(client, [query])[0]
    scores = [cosine_sim(qvec, c.embedding) for c in chunks]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top = [(c, s) for (c, s) in ranked[:top_k] if s >= SIM_THRESHOLD]
    return [c for (c, _) in top]

# ---------- Main method ----------

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    parser = argparse.ArgumentParser(description="RAG over earnings-call transcripts")
    parser.add_argument("--data", default="data", help="Directory with *.json transcripts")
    parser.add_argument("--ask", help="One-shot question about the data transcripts")
    args = parser.parse_args()

    print("Loading data and chunking...")
    chunks = build_corpus(args.data)

    print("Embedding corpus...")
    embed_corpus(client, chunks)

    def answer_query(q: str):
        retrieved = retrieve(client, chunks, q, TOP_K)
        if not retrieved:
            print("Information not found in provided transcripts.")
            return
        print("\nAnswer:")
        ans = answer_with_llm(client, q, retrieved)
        print(ans)
        print("\n----")

    if args.ask:
        answer_query(args.ask)
    else:
        print("Enter questions (Ctrl+C to exit).")
        try:
            while True:
                q = input("\nQ> ").strip()
                if not q:
                    continue
                answer_query(q)
        except KeyboardInterrupt:
            print("\nGoodbye!")

if __name__ == "__main__":
    main()

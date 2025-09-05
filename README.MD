
## How to run it

### 1) Setup (Conda recommended)
```bash
conda create -n new_env python=3.11 -y
conda activate new_env
pip install -r requirements.txt
```

### 2) Add your OpenAI key
Create a `.env` file in the repo root:
```ini
OPENAI_API_KEY=sk-your-real-key-here
```
(Keep `.env` out of version control.)

### 3) Put the data files
Place the JSON data files in `./data/`, e.g.:
```
data/
  apple_2020_q3.json
  apple_2024_q3.json
  nvidia_2023_q3.json
  nvidia_2024_q4.json
```
Each JSON is an array with one object:
```json
[
  {
    "symbol": "AAPL",
    "period": "Q3",
    "year": 2024,
    "date": "YYYY-MM-DD",
    "content": "Full transcript text..."
  }
]
```

### 4) Run
**Provide the data directory (after --data) and the query (after --ask):**
```bash
python main.py --data data --ask "What was NVIDIA's Q4 2024 revenue?"
```

---

## Design choices

### Chunking
- **Token-based chunking** with `tiktoken`:
  - `CHUNK_SIZE_TOKENS = 500`, `CHUNK_OVERLAP_TOKENS = 50` (~10% overlap).
  - These numbers proved optimal for retrieval quality and avoid cutting key phrases across chunk boundaries.

### Embeddings & Retrieval
- **Embeddings**: `text-embedding-3-small` (speed/cost/quality tradeoff is great for a small corpus).
- **Similarity**: Cosine similarity over NumPy vectors.
- **Top-K**: `TOP_K = 6` relevant candidates passed to the LLM.
- **Threshold**: `SIM_THRESHOLD = 0.18` to filter obviously off-topic chunks.

### Answer generation
- **Model**: `gpt-4o-mini` (fast and instruction-following for this scale).
- **Strict system prompt and rules**:
  - Only use provided excerpts.
  - If not answerable → reply **exactly**: `Information not found in provided transcripts.`
  - Keep answers concise; don’t add unrelated context.
  - Always include a **Sources** line.
- **Citations**: **Snippet-based** — cite the filename with a short quoted snippet from the used excerpt(s), e.g.  
  `Sources: [nvidia_2024_q4.json: "Revenue of $22.1 billion was up 22%…"]`. Snippets are used here since page/line numbers are irrelevant for the text format.

---

## Example queries and answers

### Example 1
**Q**: python main.py --data data --ask "What was NVIDIA's Q4 2024 revenue?"

**A**:
```
Answer:
NVIDIA's Q4 2024 revenue was $22.1 billion.  
Sources:  
[nvidia_2024_q4.json: "Revenue of $22.1 billion was up 22% sequentially and up to 265% year-on-year"]
```

### Example 2
**Q**: python main.py --data data --ask "What was Apple's Q3 2024 revenue?"

**A**:
```
Answer:
Apple's Q3 2024 revenue was $85.8 billion.  
Sources:  
- [apple_2024_q3.json: "new June quarter revenue record of $85.8 billion, up 5% year-over-year"]
```

### Example 3
**Q**: python main.py --data data --ask "Who loves Apple?"

**A**:
```
Answer:
Information not found in provided transcripts.  
Sources: []
```

---

## Known limitations

- **Small, heuristic retriever**: Plain cosine similarity over static embeddings. For larger corpora or more nuanced queries, could add reranking or an ANN index or use a different metric.
- **Token-window chunking**: Optimal values for this use-case but may need to be adjusted for scaling.
- **No Frontend UI**: Could create a Frontend UI; easy to extend to Flask/FastAPI if needed.

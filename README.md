#  LangGraph Job Search Agent

An educational **AI agent** that reads your CV (PDF), decides its own job searches,
pulls real job listings, ranks them against your profile, and writes a personalised
summary — all built with **LangGraph**.

Built as a teaching project: the code is deliberately small and simple so beginners
can follow every line.

---

##  What it does

```
Your CV (PDF)
   ↓  read the text
   ↓  LLM turns it into a structured profile (name, skills, experience…)
   ↓  the agent decides what jobs to search for
   ↓  calls the JSearch API → real job listings
   ↓  ranks each job by how well it matches your CV
   ↓  writes a short, encouraging summary
Top matched jobs + Apply links
```

## 🧩 The 5 steps (LangGraph nodes)

```
START → parse_cv → agent ⇄ tools → score → format → END
```

| Node | What it does |
|------|--------------|
| `parse_cv` | Reads the CV text → fills a `CVProfile` (Pydantic) using the LLM |
| `agent`    | The "brain" — decides which job searches to run |
| `tools`    | Actually calls `search_jobs` (the JSearch API) |
| `score`    | Ranks each job by local-embedding similarity to the CV |
| `format`   | Writes the final personalised summary |

The `agent ⇄ tools` loop is the **ReAct pattern**: think → act → observe → repeat.

---

##  Project files

| File | What it is |
|------|-----------|
| `job_search_agent.ipynb` | The **notebook** — build & understand the agent step by step |
| `cv_parser.py`   | Reads the PDF + turns CV text into a structured profile |
| `job_tool.py`    | The `search_jobs` tool (JSearch API + cache + sample fallback) |
| `scorer.py`      | Ranks jobs against the CV using local embeddings (fastembed) + cosine |
| `agent_core.py`  | Builds the LangGraph graph (State + 5 nodes + edges) |
| `streamlit_app.py` | The web UI (upload CV → see ranked jobs + Opik traces) |
| `sample_jobs.json` | Backup job listings shown when the live API returns nothing |
| `requirements.txt` | The Python packages to install |
| `guides/`        | **Line-by-line explanation of every file** (start here to learn) |

> 📖 **New here? Read `guides/` in order** — each file is explained line by line
> with examples.

---

##  Setup (one time)

**1. Create and activate a virtual environment**
```bash
python -m venv langraph_job
# Windows:
langraph_job\Scripts\activate
# Mac/Linux:
source langraph_job/bin/activate
```

**2. Install the packages**
```bash
pip install -r requirements.txt

pip install jupyter ipykernel
```

**3. Add your API keys**
- Copy `.env.example` → rename the copy to `.env`
- Paste your real keys inside. You need three (all have free tiers):
  - `GROQ_API_KEY` — https://console.groq.com
  - `RAPIDAPI_KEY` — https://rapidapi.com → subscribe to **JSearch**
  - `OPIK_API_KEY` — https://www.comet.com/opik

---

## ▶️ How to run

**Option A — the Notebook (to learn how it works)**
- Open `job_search_agent.ipynb` in Jupyter / VS Code
- Change the `pdf_path` in Step 2 to point at your own CV
- **Kernel → Restart & Run All Cells**

**Option B — the Web App (to demo it)**
```bash
streamlit run streamlit_app.py
```
Then upload a CV and click **Find Matching Jobs**.

---

##  Free-tier limits (important for a live demo)

This app depends on three free services with quotas. Know them before class:

| Service | Free limit | What happens when you hit it |
|---------|-----------|------------------------------|
| **JSearch** (jobs) | 200 requests / month | Returns nothing → app shows **sample jobs** instead |
| **Groq** (LLM) | ~100,000 tokens / day | `RateLimitError 429` → wait, or switch model |
| **Opik** (traces) | generous free tier | Rarely an issue |
| **Embeddings** (scoring) | free — runs **locally** (fastembed) | No quota, no key; ~130 MB model downloads once |

**Tips to survive a class demo:**
- The app **caches** successful searches (`jobs_cache.json`) — re-running the same
  search costs **0** JSearch requests.
- To run with **zero** API calls, open `job_tool.py` and set `USE_LIVE_API = False`.
  It will serve cached / sample jobs only.
- If Groq's daily tokens run out, switch the model from `llama-3.3-70b-versatile`
  to `llama-3.1-8b-instant` (higher free limits) in `cv_parser.py` and `agent_core.py`.

---

##  Where Opik (tracing) lives

- **Notebook:** kept simple — **no tracing**, it just runs and prints the answer.
- **Streamlit app:** traces every LLM call to Opik and shows a
  **"🔍 View traces in Opik"** link. Open it to see tokens, cost and latency.

---

##  Note on scoring (`scorer.py`)

Job matching uses **fastembed** (`BAAI/bge-small-en-v1.5`) to create embeddings, then
ranks jobs by cosine similarity to the CV. It runs **locally on ONNX runtime** — free,
no API key, and works offline after the model downloads once (~130 MB).

We deliberately do **not** use ChromaDB / PyTorch embeddings (`sentence-transformers`):
on Windows those caused `OSError 1455 — paging file too small`. fastembed avoids that
crash entirely, which is why those heavy libraries are no longer in `requirements.txt`.

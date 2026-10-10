# SmartSearch AI — Professional Hybrid Search Engine

**SmartSearch AI** is an advanced AI-powered hybrid search engine combining classical Information Retrieval (BM25) with modern Dense Vector Semantic Search, grounded conversational AI answers, interactive citations, and an intuitive modern UI.

---

## 🌟 Key Features

1. **Hybrid Ranking Pipeline**:
   - **BM25 Keyword Search**: Inverted index with TF-IDF weighting, phrase matching, and title boost.
   - **Semantic Vector Search**: Local sentence transformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity.
   - **Configurable Weights**: Dynamic hybrid weighting (default 50% BM25 / 50% Semantic) adjustable via `.env` or API query parameters.
   - **Cached Embeddings**: Vector embeddings are compressed and cached to disk (`indexes/embeddings.npz`) for instantaneous application startup.
2. **Intelligent Query Understanding**:
   - Intent detection (informational, questions, definitions, how-to).
   - Advanced search operators:
     - `"exact phrase"` — Prioritizes exact contiguous phrase matches.
     - `AND` — Enforces presence of both terms.
     - `OR` — Matches either term.
     - `-negation` — Excludes documents containing the negated word (e.g. `python -database`).
   - Automated typo correction with *"Showing results for..."* and *"Did you mean?"* suggestions.
   - Fast autocomplete suggestions (`/api/suggest?q=...`) covering multi-word phrases and vocabulary.
   - Contextual related search queries (`/api/related?q=...`).
3. **Grounded AI Answers & Interactive Citations**:
   - Synthesizes direct answers strictly grounded in indexed documents.
   - Interactive citation badges (`[1]`, `[2]`) that smoothly scroll to and highlight the corresponding source card.
   - Supports Google Gemini API (`GEMINI_API_KEY`), with seamless offline fallback to grounded extractive summarization.
4. **Content Quality & Deduplication**:
   - Spam and low-quality document analysis (penalizing repetitive or promotional text).
   - SHA-256 fingerprinting for duplicate document detection and penalty.
5. **Filters & Sorting**:
   - Category filtering: **All**, **Programming**, **AI**, **Networking**, **Database**, **Web Development**.
   - Sorting options: **Relevance (Hybrid)**, **Newest**, **Alphabetical (A-Z)**.
6. **Search History & Bookmarks**:
   - Search history tracking with result counts and one-click query re-run.
   - Bookmark management to save, view, and search favorite results.
   - Synchronized locally in JSON files and `localStorage`.
7. **Search Analytics**:
   - Built-in analytics dashboard tracking total searches, unique queries, zero-result searches, and click-through rates.
8. **Modern Responsive UI**:
   - Dark mode & Light mode toggle with zero-flicker (FOUC) persistence.
   - Voice search powered by the browser's Web Speech API (with animated listening feedback).
   - Quick topic chips: Python, AI, Machine Learning, Networking, Database, Web Development.
   - Full document viewer modal for reading original knowledge texts.
   - Fully responsive across desktop, tablet, and mobile browsers.

---

## 📁 Project Structure

```text
SmartSearch-AI/
│
├── main.py                     # FastAPI application & API endpoints
├── search_engine.py            # Hybrid search engine (BM25 + Vectors + Ranking)
├── ai_engine.py                # Conversational RAG & grounded answer synthesis
├── web_search.py               # Optional web search provider (DDG)
├── storage.py                  # Local persistence for history, bookmarks & analytics
├── reindex.py                  # CLI reindexing tool
│
├── data/
│   ├── documents/              # Knowledge base text files
│   │   ├── ai.txt
│   │   ├── database.txt
│   │   ├── networking.txt
│   │   ├── python.txt
│   │   ├── python_duplicate.txt
│   │   └── web_development.txt
│   ├── history.json            # Search history store
│   ├── bookmarks.json          # Bookmarks store
│   └── analytics.json          # Search analytics store
│
├── indexes/                    # Cached vector embeddings & manifests
│   ├── embeddings.npz
│   └── doc_metadata.json
│
├── tests/
│   └── test_search.py          # Automated test suite (14 test modules)
│
├── index.html                  # Modern responsive frontend
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── run.bat                     # Windows 1-click launch batch script
└── backup_pre_hybrid/          # Preserved pre-hybrid state
```

---

## 🚀 Windows 11 Quick Start

### 1. Open Windows PowerShell or Command Prompt

```powershell
cd C:\Users\ravig\OneDrive\Desktop\SmartSearch-AI
```

### 2. Activate or Create Virtual Environment

```powershell
# If creating a new virtual environment:
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1
# (or in CMD: .\venv\Scripts\activate.bat)
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

If you wish to use Google Gemini for AI answers, edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Note:** If `GEMINI_API_KEY` is not provided, SmartSearch AI automatically uses its built-in local grounded extractive answer mode. A paid API key is **never required** for basic search and answers.

### 5. Launch the Server

Run the development server with live reload:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Or simply double-click **`run.bat`** in File Explorer!

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🛠️ Management & Administration

### Adding New Knowledge Documents
1. Place any new `.txt` or `.md` files into `data/documents/`.
2. Reindex using either the CLI command or the API endpoint.

### Rebuilding the Index (CLI)
To scan documents and update vector embeddings:
```powershell
.\venv\Scripts\python.exe reindex.py
```
To force recomputing embeddings even if files haven't changed:
```powershell
.\venv\Scripts\python.exe reindex.py --force
```

### Rebuilding the Index (API)
Send a POST request to:
```http
POST /api/reindex?force=true
```

---

## 🧪 Running Automated Tests

Run the full automated test suite verifying all 14 search features:

```powershell
.\venv\Scripts\python.exe tests/test_search.py
```

All 14 test suites will run and report:
- Health check
- BM25 & Semantic vector similarity
- Hybrid ranking & configurable weights
- Duplicate document penalty
- Typo correction
- Autocomplete suggestions & related queries
- Advanced search operators (`"..."`, `AND`, `OR`, `-negation`)
- Category filtering & sorting
- Modern & legacy API endpoints
- Grounded AI answers & citations
- History & bookmarks storage
- Reindex & document viewer
- Empty & unknown query resilience

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, indexed doc count, and model status |
| `GET` | `/api/search?q={query}` | Hybrid search with BM25, semantic vectors, filters, and typo correction |
| `GET` | `/api/suggest?q={prefix}` | Autocomplete suggestions covering phrases, titles, and words |
| `GET` | `/api/related?q={query}` | Contextual related search queries |
| `GET` / `POST` | `/api/ask` | Grounded AI answer synthesis with sources and citations |
| `GET` | `/api/document/{filename}`| Safe full text viewer for indexed knowledge files |
| `POST` | `/api/reindex` | Trigger knowledge base reindexing |
| `GET` / `DELETE` | `/api/history` | View or clear search history |
| `GET` / `POST` / `DELETE`| `/api/bookmarks` | Manage bookmarked search results |
| `GET` / `POST` | `/api/analytics` | Search analytics dashboard & click tracking |
| `GET` | `/search?q={query}` | Legacy Phase 1 & 2 search route |
| `GET` | `/suggest?q={query}` | Legacy Phase 2 suggestion route |
| `GET` | `/ask?q={query}` | Legacy Phase 2 ask route |

---

## 📄 License
MIT License. Built with FastAPI, Sentence Transformers, and Google Gemini.

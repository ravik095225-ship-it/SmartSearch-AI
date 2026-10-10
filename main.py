"""
SmartSearch AI - FastAPI Application Server
Combines BM25, Semantic Vector Search, Grounded AI Answers,
Autocomplete, History, Bookmarks, and Search Analytics.
"""

import os
import ssl
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from search_engine import (
    search,
    reindex as reindex_engine,
    get_suggestions,
    get_related_searches,
    understand_query,
    DOCUMENTS,
    INDEX,
    MODEL_NAME,
    DOCUMENT_EMBEDDINGS,
    DOCUMENTS_FOLDER
)
from ai_engine import generate_answer
from web_search import search_web
import storage

LOCAL_RELEVANCE_THRESHOLD = float(os.getenv("LOCAL_RELEVANCE_THRESHOLD", "4.0"))


# =========================================================
# APP INITIALIZATION
# =========================================================

app = FastAPI(
    title="SmartSearch AI",
    description="Professional AI-Powered Hybrid Search Engine with BM25, Semantic Vectors, and Grounded Citations",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query or question")

class BookmarkRequest(BaseModel):
    title: str
    snippet: Optional[str] = ""
    content: Optional[str] = ""
    source: Optional[str] = ""
    filename: Optional[str] = ""
    category: Optional[str] = "General"
    url: Optional[str] = None
    score: Optional[float] = 0.0

class ClickRequest(BaseModel):
    title: str
    filename: Optional[str] = None
    url: Optional[str] = None


# =========================================================
# SYSTEM & HEALTH
# =========================================================

@app.get("/health")
def health_check():
    """Health check endpoint reporting system and model status."""
    return {
        "status": "healthy",
        "service": "SmartSearch AI",
        "version": "5.0.0",
        "documents_indexed": len(DOCUMENTS),
        "vocabulary_terms": len(INDEX),
        "semantic_model": MODEL_NAME,
        "vector_embeddings_cached": DOCUMENT_EMBEDDINGS is not None
    }


@app.get("/")
def home():
    """Serve the modern search frontend."""
    return FileResponse("index.html")


# =========================================================
# PROFESSIONAL HYBRID SEARCH APIS
# =========================================================

@app.get("/api/search")
def api_search(
    q: str = Query(..., description="Search query string"),
    category: Optional[str] = Query(None, description="Filter by category (e.g. Programming, AI, Database)"),
    sort: Optional[str] = Query("relevance", description="Sort by relevance, newest, or alphabetical"),
    bm25_weight: Optional[float] = Query(None, ge=0.0, le=1.0, description="BM25 scoring weight"),
    semantic_weight: Optional[float] = Query(None, ge=0.0, le=1.0, description="Semantic scoring weight"),
    include_web: bool = Query(True, description="Whether to include web results")
):
    """
    Execute hybrid search pipeline:
    BM25 + Semantic Vector search + Typo correction + Query understanding + Related queries.
    """
    clean_q = q.strip()
    if not clean_q:
        return {
            "query": "",
            "corrected_query": None,
            "did_you_mean": None,
            "understanding": {},
            "total": 0,
            "results": [],
            "web_results": [],
            "related_searches": []
        }

    # Query understanding & typo detection
    understanding = understand_query(clean_q)
    did_you_mean = understanding.get("did_you_mean")

    # Local Hybrid Search
    local_results = search(
        clean_q,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        category=category,
        sort_by=sort or "relevance"
    )

    # If typo detected and local results are sparse, search using corrected query
    if did_you_mean and did_you_mean.lower() != clean_q.lower() and len(local_results) == 0:
        corrected_results = search(
            did_you_mean,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            category=category,
            sort_by=sort or "relevance"
        )
        if corrected_results:
            local_results = corrected_results

    # Check if local results are sufficient
    max_local_score = max((r.get("score", 0.0) for r in local_results), default=0.0)
    has_sufficient_local = bool(local_results) and max_local_score >= LOCAL_RELEVANCE_THRESHOLD

    # Web search fallback & combination
    web_results = []
    should_fetch_web = include_web and (not has_sufficient_local or (category and category.lower() == "web"))
    if category and category.lower() not in ("all", "web") and local_results:
        should_fetch_web = False

    if should_fetch_web:
        try:
            web_results = search_web(clean_q, max_results=8)
        except Exception as e:
            print(f"[API SEARCH] Web search error: {e}")
            web_results = []

    # Determine final results and search mode
    search_mode = "local"
    if not local_results and web_results:
        final_results = web_results
        search_mode = "web_fallback"
    elif local_results and not has_sufficient_local and web_results:
        final_results = local_results + web_results
        search_mode = "combined"
    else:
        final_results = local_results
        search_mode = "local"

    # Related searches
    top_cat = final_results[0]["category"] if final_results else understanding.get("topic")
    related = get_related_searches(clean_q, top_category=top_cat)

    # Record analytics and history with final_results count
    storage.record_search(clean_q, len(final_results))
    storage.add_history(clean_q, len(final_results))

    return {
        "query": clean_q,
        "corrected_query": did_you_mean,
        "did_you_mean": did_you_mean,
        "understanding": understanding,
        "total": len(final_results),
        "results": final_results,
        "local_results": local_results,
        "web_results": web_results,
        "search_mode": search_mode,
        "related_searches": related,
        "category": category or "All",
        "sort": sort or "relevance"
    }


@app.get("/api/suggest")
def api_suggest(q: str = Query("", description="Query prefix for autocomplete")):
    """Autocomplete suggestions covering phrases, titles, and vocabulary."""
    suggestions = get_suggestions(q, limit=8)
    return {
        "query": q,
        "suggestions": suggestions
    }


@app.get("/api/related")
def api_related(q: str = Query(..., description="Query for related searches")):
    """Contextual related search suggestions."""
    related = get_related_searches(q)
    return {
        "query": q,
        "related_searches": related
    }


@app.get("/api/ask")
def api_ask_get(q: str = Query(..., description="Question to answer using grounded RAG")):
    """Generate grounded AI answer using local documents with citation markers."""
    result = generate_answer(q)
    return result


@app.post("/api/ask")
def api_ask_post(req: AskRequest):
    """Generate grounded AI answer using local documents with citation markers."""
    result = generate_answer(req.query)
    return result


@app.post("/api/reindex")
def api_reindex(force: bool = Query(True, description="Force recompute embeddings")):
    """Trigger reindexing of knowledge documents and update vector cache."""
    result = reindex_engine(force=force)
    return result


@app.get("/api/document/{filename}")
def api_get_document(filename: str):
    """Safely fetch full content of an indexed knowledge document."""
    # Prevent directory traversal
    clean_name = os.path.basename(filename)
    file_path = DOCUMENTS_FOLDER / clean_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        content = file_path.read_text(encoding="utf-8")
        title = clean_name.replace(".txt", "").replace("_", " ").title()
        return {
            "title": title,
            "filename": clean_name,
            "content": content,
            "size_bytes": file_path.stat().st_size,
            "modified_time": file_path.stat().st_mtime
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# HISTORY & BOOKMARKS APIS
# =========================================================

@app.get("/api/history")
def get_search_history(limit: int = Query(50, ge=1, le=100)):
    """Retrieve recent search history."""
    return {
        "history": storage.get_history(limit=limit)
    }


@app.delete("/api/history/{item_id}")
def delete_search_history_item(item_id: str):
    """Delete a specific history entry."""
    success = storage.delete_history_item(item_id)
    return {"success": success}


@app.delete("/api/history")
def clear_all_history():
    """Clear all search history."""
    storage.clear_history()
    return {"success": True, "message": "Search history cleared."}


@app.get("/api/bookmarks")
def get_user_bookmarks():
    """Retrieve bookmarked items."""
    return {
        "bookmarks": storage.get_bookmarks()
    }


@app.post("/api/bookmarks")
def add_user_bookmark(item: BookmarkRequest):
    """Add a new bookmark."""
    saved = storage.add_bookmark(item.dict())
    return {"success": True, "bookmark": saved}


@app.delete("/api/bookmarks/{item_id}")
def delete_user_bookmark(item_id: str):
    """Remove a bookmark by ID or filename."""
    success = storage.remove_bookmark(item_id)
    return {"success": success}


# =========================================================
# ANALYTICS APIS
# =========================================================

@app.get("/api/analytics")
def get_analytics_data():
    """Get search analytics, popular queries, and click-through metrics."""
    return storage.get_analytics()


@app.post("/api/analytics/click")
def record_result_click(click: ClickRequest):
    """Record a user click on a search result."""
    storage.record_click(title=click.title, filename=click.filename, url=click.url)
    return {"success": True}


# =========================================================
# IN-APP WEBSITE VIEWER APIS
# =========================================================

@app.get("/api/check-frame")
def api_check_frame(url: str = Query(..., description="Target website URL")):
    """
    Check if a website allows iframe embedding by inspecting X-Frame-Options and CSP headers.
    Validates URL scheme (strictly http / https only) and does not proxy content.
    """
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL scheme. Only HTTP and HTTPS URLs are permitted."
        )

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url.strip(),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 SmartSearch/5.0"
        }
    )
    req.get_method = lambda: "HEAD"

    try:
        with urllib.request.urlopen(req, timeout=2.5, context=ctx) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            xfo = headers.get("x-frame-options", "").upper()
            csp = headers.get("content-security-policy", "").lower()

            if "DENY" in xfo:
                return {
                    "url": url,
                    "domain": parsed.netloc,
                    "can_embed": False,
                    "reason": "Website blocks embedding via 'X-Frame-Options: DENY'",
                    "x_frame_options": xfo
                }
            if "SAMEORIGIN" in xfo:
                return {
                    "url": url,
                    "domain": parsed.netloc,
                    "can_embed": False,
                    "reason": "Website blocks embedding via 'X-Frame-Options: SAMEORIGIN'",
                    "x_frame_options": xfo
                }
            if "frame-ancestors" in csp and ("'none'" in csp or "'self'" in csp):
                return {
                    "url": url,
                    "domain": parsed.netloc,
                    "can_embed": False,
                    "reason": "Website blocks embedding via Content-Security-Policy (frame-ancestors)",
                    "content_security_policy": csp
                }

            return {
                "url": url,
                "domain": parsed.netloc,
                "can_embed": True,
                "reason": "No iframe embedding restrictions detected"
            }
    except Exception as e:
        # If HEAD check times out or site returns error on HEAD, allow client iframe to attempt direct load
        return {
            "url": url,
            "domain": parsed.netloc,
            "can_embed": True,
            "reason": f"Header check inconclusive ({str(e)}), attempting direct iframe display"
        }


# =========================================================
# BACKWARD-COMPATIBILITY ROUTES (Phase 1 & Phase 2)
# =========================================================

@app.get("/search")
def search_legacy(q: str):
    """Preserve existing /search API contract with automatic web fallback."""
    local_results = search(q)
    max_local_score = max((r.get("score", 0.0) for r in local_results), default=0.0)
    has_sufficient_local = bool(local_results) and max_local_score >= LOCAL_RELEVANCE_THRESHOLD
    web_results = []

    if not has_sufficient_local:
        try:
            web_results = search_web(q, max_results=8)
        except Exception as e:
            print(f"[SEARCH LEGACY] Web search error: {e}")
            web_results = []

    if not local_results and web_results:
        final_results = web_results
    elif local_results and not has_sufficient_local and web_results:
        final_results = local_results + web_results
    else:
        final_results = local_results

    storage.record_search(q, len(final_results))
    storage.add_history(q, len(final_results))

    return {
        "query": q,
        "local_results": local_results,
        "web_results": web_results,
        "results": final_results
    }


@app.get("/web-search")
def web_search_legacy(q: str):
    """Preserve existing /web-search API contract."""
    results = search_web(q, max_results=8)
    return {
        "query": q,
        "results": results
    }


@app.get("/suggest")
def suggest_legacy(q: str):
    """Preserve and enhance existing /suggest API contract."""
    clean_q = q.strip().lower()
    if not clean_q:
        return {"query": clean_q, "suggestions": []}

    suggestions = get_suggestions(clean_q, limit=8)
    return {
        "query": clean_q,
        "suggestions": suggestions
    }


@app.get("/ask")
def ask_ai_legacy(q: str):
    """Preserve existing /ask API contract."""
    return generate_answer(q)


@app.post("/reindex")
def reindex_legacy():
    """Preserve /reindex shortcut."""
    return reindex_engine(force=True)
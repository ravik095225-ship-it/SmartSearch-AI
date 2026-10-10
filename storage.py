"""
SmartSearch AI - Persistent Storage & Analytics
Manages search history, bookmarks, and search analytics in local JSON files.
"""

from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = DATA_DIR / "history.json"
BOOKMARKS_FILE = DATA_DIR / "bookmarks.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"


# =========================================================
# HELPER JSON FUNCTIONS
# =========================================================

def _read_json(file_path: Path, default: Any) -> Any:
    if not file_path.exists():
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(file_path: Path, data: Any) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[STORAGE ERROR] Could not write to {file_path.name}: {e}")


# =========================================================
# SEARCH HISTORY
# =========================================================

def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    history = _read_json(HISTORY_FILE, [])
    # Return newest first
    return sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]


def add_history(query: str, result_count: int = 0) -> Dict[str, Any]:
    q = query.strip()
    if not q:
        return {}

    history = _read_json(HISTORY_FILE, [])

    # Remove existing duplicate query to push to top
    history = [item for item in history if item.get("query", "").lower() != q.lower()]

    item = {
        "id": f"hist_{int(time.time() * 1000)}",
        "query": q,
        "result_count": result_count,
        "timestamp": int(time.time()),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    history.insert(0, item)
    history = history[:100]  # keep last 100
    _write_json(HISTORY_FILE, history)
    return item


def delete_history_item(item_id: str) -> bool:
    history = _read_json(HISTORY_FILE, [])
    initial_len = len(history)
    history = [item for item in history if item.get("id") != item_id]
    if len(history) != initial_len:
        _write_json(HISTORY_FILE, history)
        return True
    return False


def clear_history() -> bool:
    _write_json(HISTORY_FILE, [])
    return True


# =========================================================
# BOOKMARKS
# =========================================================

def get_bookmarks() -> List[Dict[str, Any]]:
    bookmarks = _read_json(BOOKMARKS_FILE, [])
    return sorted(bookmarks, key=lambda x: x.get("timestamp", 0), reverse=True)


def add_bookmark(bookmark_data: Dict[str, Any]) -> Dict[str, Any]:
    title = bookmark_data.get("title", "").strip()
    if not title:
        return {}

    bookmarks = _read_json(BOOKMARKS_FILE, [])
    
    # Check if already bookmarked
    url = bookmark_data.get("url") or bookmark_data.get("filename") or title
    for b in bookmarks:
        existing_target = b.get("url") or b.get("filename") or b.get("title")
        if existing_target == url:
            return b  # already exists

    item = {
        "id": f"bm_{int(time.time() * 1000)}",
        "title": title,
        "snippet": bookmark_data.get("snippet", "")[:300],
        "content": bookmark_data.get("content", ""),
        "source": bookmark_data.get("source") or bookmark_data.get("filename", ""),
        "filename": bookmark_data.get("filename", ""),
        "category": bookmark_data.get("category", "General"),
        "url": bookmark_data.get("url"),
        "score": bookmark_data.get("score", 0),
        "timestamp": int(time.time()),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    bookmarks.insert(0, item)
    _write_json(BOOKMARKS_FILE, bookmarks)
    return item


def remove_bookmark(item_id: str) -> bool:
    bookmarks = _read_json(BOOKMARKS_FILE, [])
    initial_len = len(bookmarks)
    # Check match by id or filename or title
    bookmarks = [
        b for b in bookmarks 
        if b.get("id") != item_id and b.get("filename") != item_id and b.get("title") != item_id
    ]
    if len(bookmarks) != initial_len:
        _write_json(BOOKMARKS_FILE, bookmarks)
        return True
    return False


# =========================================================
# SEARCH ANALYTICS
# =========================================================

def get_analytics() -> Dict[str, Any]:
    analytics = _read_json(ANALYTICS_FILE, {
        "total_searches": 0,
        "query_counts": {},
        "zero_result_searches": 0,
        "total_results_served": 0,
        "clicks": []
    })

    total_searches = analytics.get("total_searches", 0)
    query_counts: Dict[str, int] = analytics.get("query_counts", {})
    zero_searches = analytics.get("zero_result_searches", 0)
    total_results = analytics.get("total_results_served", 0)
    clicks = analytics.get("clicks", [])

    # Top queries
    sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
    top_queries = [
        {"query": q, "count": cnt}
        for q, cnt in sorted_queries[:10]
    ]

    # Top clicked items
    click_counts: Dict[str, int] = {}
    for c in clicks:
        title = c.get("title", "Unknown")
        click_counts[title] = click_counts.get(title, 0) + 1
    
    sorted_clicks = sorted(click_counts.items(), key=lambda x: x[1], reverse=True)
    top_clicked = [
        {"title": title, "clicks": cnt}
        for title, cnt in sorted_clicks[:10]
    ]

    avg_results = round(total_results / max(1, total_searches), 1)

    return {
        "total_searches": total_searches,
        "unique_queries": len(query_counts),
        "zero_result_searches": zero_searches,
        "avg_results_per_search": avg_results,
        "top_queries": top_queries,
        "top_clicked": top_clicked,
        "total_clicks": len(clicks)
    }


def record_search(query: str, result_count: int) -> None:
    q = query.strip().lower()
    if not q:
        return

    analytics = _read_json(ANALYTICS_FILE, {
        "total_searches": 0,
        "query_counts": {},
        "zero_result_searches": 0,
        "total_results_served": 0,
        "clicks": []
    })

    analytics["total_searches"] = analytics.get("total_searches", 0) + 1
    analytics["total_results_served"] = analytics.get("total_results_served", 0) + result_count
    
    if result_count == 0:
        analytics["zero_result_searches"] = analytics.get("zero_result_searches", 0) + 1

    qc = analytics.get("query_counts", {})
    qc[q] = qc.get(q, 0) + 1
    analytics["query_counts"] = qc

    _write_json(ANALYTICS_FILE, analytics)


def record_click(title: str, filename: Optional[str] = None, url: Optional[str] = None) -> None:
    analytics = _read_json(ANALYTICS_FILE, {
        "total_searches": 0,
        "query_counts": {},
        "zero_result_searches": 0,
        "total_results_served": 0,
        "clicks": []
    })

    clicks = analytics.get("clicks", [])
    clicks.append({
        "title": title,
        "filename": filename,
        "url": url,
        "timestamp": int(time.time())
    })
    analytics["clicks"] = clicks[-200:]  # keep recent 200 clicks

    _write_json(ANALYTICS_FILE, analytics)

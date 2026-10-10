"""
SmartSearch AI - Search Engine
Phase 3 & Phase 4 Hybrid Search Implementation:
- BM25 Inverted Index & Scoring
- Semantic Vector Search (Sentence Transformers + Cosine Similarity)
- Cached Embeddings (Fast Startup)
- Hybrid Ranking with Configurable Weights (BM25 + Semantic)
- Advanced Search Operators ("exact phrase", AND, OR, -negation)
- Quality & Spam Scoring + Duplicate Detection
- Query Understanding, Typo Correction & Related Searches
- Enhanced Autocomplete Suggestions
- Category Filtering & Sorting
- Dynamic Reindexing Support
"""

import os
from pathlib import Path
import re
import math
import hashlib
import json
from difflib import get_close_matches
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# SETTINGS & CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).parent.resolve()
DOCUMENTS_FOLDER = BASE_DIR / "data" / "documents"
INDEXES_DIR = BASE_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

CACHE_EMBEDDINGS_FILE = INDEXES_DIR / "embeddings.npz"
CACHE_METADATA_FILE = INDEXES_DIR / "doc_metadata.json"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# Configurable hybrid weights (default 50% BM25, 50% Semantic)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.50"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.50"))


# =========================================================
# STOPWORDS & VOCABULARY
# =========================================================

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "as", "by",
    "this", "that", "are", "was", "be", "from",
    "it", "at", "which", "how", "what", "why",
    "when", "where", "can", "could", "should", "would",
    "do", "does", "did", "have", "has", "had"
}

CATEGORY_MAP = {
    "ai.txt": "AI",
    "database.txt": "Database",
    "networking.txt": "Networking",
    "python.txt": "Programming",
    "python_duplicate.txt": "Programming",
    "web_development.txt": "Web Development"
}

CATEGORY_KEYWORDS = {
    "AI": ["artificial intelligence", "machine learning", "neural networks", "nlp", "deep learning", "ai", "model"],
    "Database": ["database", "sql", "relational", "tables", "rows", "columns", "query", "storage"],
    "Networking": ["networking", "network", "tcp", "ip", "http", "dns", "protocol", "packets", "router"],
    "Programming": ["python", "programming", "code", "software", "functions", "oop", "syntax", "language"],
    "Web Development": ["web development", "html", "css", "javascript", "frontend", "backend", "servers", "apis"]
}

POPULAR_SUGGESTIONS = [
    "Python programming",
    "Python data science",
    "Python machine learning",
    "Python functions",
    "Python OOP",
    "Artificial intelligence",
    "Machine learning models",
    "Neural networks",
    "Computer networking",
    "TCP IP protocols",
    "HTTP and DNS",
    "Database management",
    "Relational databases",
    "SQL queries",
    "Web development",
    "HTML CSS JavaScript",
    "Frontend development",
    "Backend servers and APIs"
]


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out common stopwords from token list."""
    return [token for token in tokens if token not in STOPWORDS]


def normalize_text(text: str) -> str:
    """Normalize text into single-spaced lowercase alphanumeric tokens."""
    return " ".join(tokenize(text))


def fingerprint(text: str) -> str:
    """Generate SHA-256 fingerprint of normalized text for deduplication."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def detect_category(filename: str, text: str) -> str:
    """Detect category using filename mapping or content keywords."""
    if filename in CATEGORY_MAP:
        return CATEGORY_MAP[filename]
    
    text_lower = text.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            return cat
    return "General"


def extract_snippet(content: str, query_terms: List[str], max_length: int = 180) -> str:
    """Extract a representative snippet around matching query terms."""
    if not content:
        return ""
    
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    if not sentences:
        return content[:max_length]
    
    best_sentence = sentences[0]
    best_match_count = -1
    
    clean_terms = [t.lower() for t in query_terms if len(t) > 2]
    
    for sentence in sentences:
        s_lower = sentence.lower()
        matches = sum(1 for t in clean_terms if t in s_lower)
        if matches > best_match_count:
            best_match_count = matches
            best_sentence = sentence
    
    if len(best_sentence) > max_length:
        return best_sentence[:max_length].rstrip() + "..."
    return best_sentence


# =========================================================
# QUALITY + SPAM ANALYSIS
# =========================================================

def analyze_quality(text: str) -> Dict[str, Any]:
    """Analyze document quality and spam signals."""
    tokens = tokenize(text)

    if not tokens:
        return {
            "quality_score": 0,
            "spam_score": 100,
            "is_low_quality": True,
            "reasons": ["Empty document"]
        }

    score = 100
    spam_score = 0
    reasons = []
    word_count = len(tokens)

    # Document length
    if word_count < 15:
        score -= 35
        spam_score += 25
        reasons.append("Very short content")
    elif word_count < 30:
        score -= 15
        spam_score += 10
        reasons.append("Short content")

    # Word repetition
    frequencies = {}
    for word in tokens:
        frequencies[word] = frequencies.get(word, 0) + 1

    most_common = max(frequencies.values()) if frequencies else 0

    if word_count >= 10:
        repetition_ratio = most_common / word_count
        if repetition_ratio > 0.50:
            score -= 25
            spam_score += 35
            reasons.append("Excessive word repetition")
        elif repetition_ratio > 0.35:
            score -= 10
            spam_score += 15
            reasons.append("High word repetition")

    # Repeated phrases
    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    ]
    if bigrams:
        phrase_counts = {}
        for phrase in bigrams:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        repeated_phrase = max(phrase_counts.values()) if phrase_counts else 0
        if repeated_phrase >= 4:
            score -= 20
            spam_score += 30
            reasons.append("Repeated phrases detected")

    # Promotional symbols
    symbol_count = len(re.findall(r"[!$%*@#]", text))
    if symbol_count > 10:
        score -= 15
        spam_score += 20
        reasons.append("Excessive promotional symbols")

    # Uppercase detection
    letters = re.findall(r"[a-zA-Z]", text)
    if len(letters) > 20:
        uppercase = sum(1 for char in letters if char.isupper())
        uppercase_ratio = uppercase / len(letters)
        if uppercase_ratio > 0.60:
            score -= 20
            spam_score += 25
            reasons.append("Excessive uppercase text")

    score = max(0, min(100, score))
    spam_score = max(0, min(100, spam_score))

    return {
        "quality_score": score,
        "spam_score": spam_score,
        "is_low_quality": score < 50,
        "reasons": reasons
    }


# =========================================================
# DOCUMENT LOADING & INDEX BUILDING
# =========================================================

def build_index() -> List[Dict[str, Any]]:
    """Scan and load all documents from the documents folder."""
    documents = []

    if not DOCUMENTS_FOLDER.exists():
        DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
        return documents

    for file_path in sorted(DOCUMENTS_FOLDER.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue

        if not text:
            continue

        title = file_path.stem.replace("_", " ").title()
        tokens = remove_stopwords(tokenize(text))
        all_tokens = tokenize(text)
        quality = analyze_quality(text)
        category = detect_category(file_path.name, text)
        stat = file_path.stat()

        documents.append({
            "title": title,
            "filename": file_path.name,
            "category": category,
            "content": text,
            "tokens": tokens,
            "all_tokens": all_tokens,
            "length": len(tokens),
            "fingerprint": fingerprint(text),
            "quality_score": quality["quality_score"],
            "spam_score": quality["spam_score"],
            "is_low_quality": quality["is_low_quality"],
            "quality_reasons": quality["reasons"],
            "is_duplicate": False,
            "duplicate_of": None,
            "modified_time": stat.st_mtime
        })

    # Duplicate detection
    seen = {}
    for document in documents:
        fp = document["fingerprint"]
        if fp in seen:
            original = seen[fp]
            document["is_duplicate"] = True
            document["duplicate_of"] = original["filename"]
            document["spam_score"] = min(100, document["spam_score"] + 25)
        else:
            seen[fp] = document

    return documents


def build_inverted_index(documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build in-memory inverted index for fast keyword lookup."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for document in documents:
        for word in set(document["tokens"]):
            if word not in index:
                index[word] = []
            index[word].append(document)
    return index


# =========================================================
# GLOBAL STATE & INITIAL LOAD
# =========================================================

print("\n" + "=" * 60)
print("SMARTSEARCH AI - HYBRID SEARCH ENGINE")
print("Initializing document index...")
print("=" * 60)

DOCUMENTS: List[Dict[str, Any]] = build_index()
INDEX: Dict[str, List[Dict[str, Any]]] = build_inverted_index(DOCUMENTS)

print(f"Documents indexed: {len(DOCUMENTS)}")


# =========================================================
# SEMANTIC MODEL & EMBEDDINGS CACHE
# =========================================================

SEMANTIC_MODEL: Optional[SentenceTransformer] = None

try:
    print("Loading semantic embedding model...")
    SEMANTIC_MODEL = SentenceTransformer(MODEL_NAME)
    print(f"Semantic model loaded: {MODEL_NAME}")
except Exception as error:
    SEMANTIC_MODEL = None
    print(f"WARNING: Semantic model could not be loaded: {error}")


def load_or_create_embeddings(
    documents: List[Dict[str, Any]],
    model: Optional[SentenceTransformer],
    force_recompute: bool = False
) -> Optional[np.ndarray]:
    """
    Load precomputed embeddings from cache if document fingerprints match,
    otherwise compute and cache them to disk for fast startup.
    """
    if model is None or not documents:
        return None

    current_manifest = {
        doc["filename"]: {
            "fingerprint": doc["fingerprint"],
            "mtime": doc["modified_time"]
        }
        for doc in documents
    }

    # Check cache validity
    if not force_recompute and CACHE_EMBEDDINGS_FILE.exists() and CACHE_METADATA_FILE.exists():
        try:
            with open(CACHE_METADATA_FILE, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            
            if cached_manifest == current_manifest:
                data = np.load(CACHE_EMBEDDINGS_FILE)
                embeddings = data["embeddings"]
                if len(embeddings) == len(documents):
                    print("Document embeddings loaded from cache (fast startup).")
                    return embeddings
        except Exception as e:
            print(f"Embeddings cache invalid ({e}), recomputing...")

    # Compute new embeddings
    try:
        print("Computing semantic embeddings for documents...")
        document_texts = [doc["content"] for doc in documents]
        embeddings = model.encode(document_texts, show_progress_bar=False)

        # Save to cache
        np.savez_compressed(CACHE_EMBEDDINGS_FILE, embeddings=embeddings)
        with open(CACHE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(current_manifest, f, indent=2)

        print("Document embeddings computed and cached successfully.")
        return embeddings
    except Exception as e:
        print(f"Failed to compute embeddings: {e}")
        return None


DOCUMENT_EMBEDDINGS: Optional[np.ndarray] = load_or_create_embeddings(DOCUMENTS, SEMANTIC_MODEL)


# =========================================================
# BM25 SCORING
# =========================================================

def document_frequency(term: str) -> int:
    return sum(1 for document in DOCUMENTS if term in document["tokens"])


def bm25_idf(term: str) -> float:
    total_documents = len(DOCUMENTS)
    df = document_frequency(term)

    if total_documents == 0:
        return 0.0

    return math.log(
        1.0 + (total_documents - df + 0.5) / (df + 0.5)
    )


def bm25_score(query_terms: List[str], document: Dict[str, Any]) -> float:
    score = 0.0
    document_length = document["length"]
    average_length = (
        sum(doc["length"] for doc in DOCUMENTS) / len(DOCUMENTS)
        if DOCUMENTS else 1.0
    )

    frequencies: Dict[str, int] = {}
    for token in document["tokens"]:
        frequencies[token] = frequencies.get(token, 0) + 1

    for term in query_terms:
        if term not in frequencies:
            continue

        tf = frequencies[term]
        numerator = tf * (BM25_K1 + 1.0)
        denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (document_length / average_length))

        score += bm25_idf(term) * (numerator / denominator)

    return score


# =========================================================
# ADVANCED QUERY PARSING & OPERATORS
# =========================================================

def parse_advanced_query(raw_query: str) -> Dict[str, Any]:
    """
    Parse search operators:
    - "exact phrase"
    - AND
    - OR
    - -negation (must NOT contain)
    """
    query = raw_query.strip()
    
    # 1. Extract exact phrases
    exact_phrases = re.findall(r'"([^"]+)"', query)
    cleaned_query = re.sub(r'"[^"]+"', ' ', query)
    
    # 2. Extract negation (-word)
    negations = [
        match.group(1).lower()
        for match in re.finditer(r'(?:^|\s)-([a-zA-Z0-9]+)', cleaned_query)
    ]
    cleaned_query = re.sub(r'(?:^|\s)-[a-zA-Z0-9]+', ' ', cleaned_query)

    # 3. Detect boolean operators
    tokens_raw = cleaned_query.split()
    must_have: List[str] = []
    should_have: List[str] = []
    has_and = "AND" in tokens_raw
    has_or = "OR" in tokens_raw

    clean_words = []
    for token in tokens_raw:
        if token in ("AND", "OR"):
            continue
        cleaned = re.sub(r"[^\w]", "", token).lower()
        if cleaned:
            clean_words.append(cleaned)

    if has_and:
        must_have.extend(remove_stopwords(clean_words))
    else:
        should_have.extend(clean_words)

    # Include exact phrase tokens in should_have for scoring
    for phrase in exact_phrases:
        should_have.extend(remove_stopwords(tokenize(phrase)))

    query_terms = remove_stopwords(tokenize(cleaned_query)) + [
        t for phrase in exact_phrases for t in remove_stopwords(tokenize(phrase))
    ]

    return {
        "raw_query": raw_query,
        "exact_phrases": exact_phrases,
        "negations": negations,
        "must_have": must_have,
        "should_have": should_have,
        "query_terms": list(dict.fromkeys(query_terms)),
        "is_advanced": bool(exact_phrases or negations or has_and or has_or)
    }


# =========================================================
# QUERY UNDERSTANDING & TYPO CORRECTION
# =========================================================

def correct_typos(query_terms: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Correct typos using fuzzy distance against index vocabulary.
    Returns: (corrected_terms, human_readable_corrections, full_corrected_query)
    """
    vocabulary = list(INDEX.keys())
    corrected = []
    corrections = []
    has_change = False

    for term in query_terms:
        if term in INDEX or len(term) <= 2:
            corrected.append(term)
            continue

        matches = get_close_matches(term, vocabulary, n=1, cutoff=0.75)
        if matches and matches[0] != term:
            corrected.append(matches[0])
            corrections.append(f"{term} → {matches[0]}")
            has_change = True
        else:
            corrected.append(term)

    did_you_mean = " ".join(corrected) if has_change else None
    return corrected, corrections, did_you_mean


def understand_query(query: str) -> Dict[str, Any]:
    """Detect keywords, phrases, topic, intent, and spelling errors."""
    q_clean = query.strip()
    words = tokenize(q_clean)
    keywords = remove_stopwords(words)
    parsed_ops = parse_advanced_query(q_clean)
    corrected_terms, corrections, did_you_mean = correct_typos(keywords)

    # Detect intent
    q_lower = q_clean.lower()
    intent = "search"
    if q_lower.startswith(("what is", "define", "definition of", "who is")):
        intent = "definition"
    elif q_lower.startswith(("how to", "how do", "how can", "steps to")):
        intent = "how-to"
    elif "?" in q_clean or q_lower.startswith(("why", "where", "which", "is", "are")):
        intent = "question"

    # Detect topic
    topic = "General"
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q_lower for kw in kws):
            topic = cat
            break

    return {
        "original_query": q_clean,
        "keywords": keywords,
        "intent": intent,
        "topic": topic,
        "parsed_operators": parsed_ops,
        "corrections": corrections,
        "did_you_mean": did_you_mean,
        "is_question": intent in ("question", "definition", "how-to")
    }


# =========================================================
# SEMANTIC SCORES
# =========================================================

def semantic_scores(query: str) -> List[float]:
    """Compute cosine similarity of query against all indexed documents."""
    if SEMANTIC_MODEL is None or DOCUMENT_EMBEDDINGS is None or not DOCUMENTS:
        return [0.0] * len(DOCUMENTS)

    try:
        query_embedding = SEMANTIC_MODEL.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, DOCUMENT_EMBEDDINGS)[0]
        return [float(score) for score in similarities]
    except Exception as e:
        print(f"Error computing semantic scores: {e}")
        return [0.0] * len(DOCUMENTS)


# =========================================================
# AUTOCOMPLETE & RELATED SEARCHES
# =========================================================

def get_suggestions(prefix: str, limit: int = 8) -> List[str]:
    """
    Return smart autocomplete suggestions:
    - Multi-word phrases matching prefix
    - Document titles
    - Indexed vocabulary tokens
    """
    p = prefix.strip().lower()
    if not p:
        return POPULAR_SUGGESTIONS[:limit]

    matches: List[str] = []

    # 1. Match against popular phrases
    for phrase in POPULAR_SUGGESTIONS:
        if phrase.lower().startswith(p):
            matches.append(phrase)

    # 2. Match document titles
    for doc in DOCUMENTS:
        title = doc["title"]
        if title.lower().startswith(p) and title not in matches:
            matches.append(title)

    # 3. Match against vocabulary
    for word in sorted(INDEX.keys()):
        if word.startswith(p):
            # Format nicely
            capitalized = word.capitalize()
            if capitalized not in matches and word not in matches:
                matches.append(capitalized)
        if len(matches) >= limit * 2:
            break

    # 4. Infix search if still sparse
    if len(matches) < limit:
        for phrase in POPULAR_SUGGESTIONS:
            if p in phrase.lower() and phrase not in matches:
                matches.append(phrase)

    unique_matches = list(dict.fromkeys(matches))
    return unique_matches[:limit]


def get_related_searches(query: str, top_category: Optional[str] = None) -> List[str]:
    """Generate contextual related search suggestions."""
    q_lower = query.lower()
    related = []

    # Category-based related queries
    topic_related = {
        "Programming": [
            "Python programming basics",
            "Python data science and AI",
            "Python functions and OOP",
            "Python automation scripts",
            "Python vs other languages"
        ],
        "AI": [
            "Machine learning fundamentals",
            "Neural networks and deep learning",
            "Natural language processing",
            "AI algorithms and models",
            "Supervised vs unsupervised learning"
        ],
        "Networking": [
            "Computer networking protocols",
            "TCP vs IP difference",
            "HTTP DNS packet flow",
            "Reliable communication in networking",
            "Network devices and routers"
        ],
        "Database": [
            "Relational database management",
            "SQL query examples",
            "Database tables rows columns",
            "CRUD operations in SQL",
            "Relational vs NoSQL databases"
        ],
        "Web Development": [
            "HTML CSS JavaScript fundamentals",
            "Frontend vs backend development",
            "Web APIs and server communication",
            "Building responsive web applications",
            "Modern JavaScript frameworks"
        ]
    }

    # Match topic
    cat = top_category
    if not cat or cat == "General":
        for k, kws in CATEGORY_KEYWORDS.items():
            if any(kw in q_lower for kw in kws):
                cat = k
                break

    if cat and cat in topic_related:
        for r in topic_related[cat]:
            if r.lower() != q_lower:
                related.append(r)

    # If query matches specific terms
    if "python" in q_lower:
        related.extend([
            "Python for loops and while loops",
            "Python data structures",
            "Python web frameworks"
        ])
    elif "network" in q_lower or "tcp" in q_lower:
        related.extend([
            "OSI model 7 layers",
            "TCP three way handshake"
        ])
    elif "sql" in q_lower or "database" in q_lower:
        related.extend([
            "SQL join types",
            "Database indexing methods"
        ])

    unique_related = list(dict.fromkeys(related))
    return unique_related[:5] if unique_related else [
        "Python programming",
        "Machine learning basics",
        "Computer networking",
        "SQL databases",
        "Web development"
    ]


# =========================================================
# MAIN HYBRID SEARCH
# =========================================================

def search(
    query: str,
    bm25_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    category: Optional[str] = None,
    sort_by: str = "relevance"
) -> List[Dict[str, Any]]:
    """
    Perform full hybrid search combining BM25 keyword matching and vector semantic search.
    Supports advanced operators, configurable weights, typo correction, filtering, and sorting.
    """
    query = query.strip()
    if not query:
        return []

    # Configurable weights
    w_bm25 = bm25_weight if bm25_weight is not None else BM25_WEIGHT
    w_semantic = semantic_weight if semantic_weight is not None else SEMANTIC_WEIGHT

    # Normalize weights if sum is non-zero
    w_total = w_bm25 + w_semantic
    if w_total > 0:
        w_bm25 /= w_total
        w_semantic /= w_total

    # Query understanding & parsing
    parsed_ops = parse_advanced_query(query)
    original_terms = parsed_ops["query_terms"]

    if not original_terms and not parsed_ops["exact_phrases"]:
        return []

    # Typo correction
    query_terms, corrections, did_you_mean = correct_typos(original_terms)

    # Candidate selection
    candidate_documents = []
    for document in DOCUMENTS:
        # Category filter check
        if category and category.lower() != "all" and document["category"].lower() != category.lower():
            continue

        # Advanced operator check: Negations (-term)
        if parsed_ops["negations"]:
            doc_all_tokens = set(document["all_tokens"])
            if any(neg in doc_all_tokens for neg in parsed_ops["negations"]):
                continue

        # Advanced operator check: Must Have (AND)
        if parsed_ops["must_have"]:
            doc_tokens = set(document["tokens"])
            if not all(term in doc_tokens for term in parsed_ops["must_have"]):
                continue

        # Normal candidate matching: check if document contains any query term or phrase
        norm_content = normalize_text(document["content"])
        has_phrase_match = any(
            normalize_text(phrase) in norm_content
            for phrase in parsed_ops["exact_phrases"]
        )

        has_term_match = any(term in document["tokens"] for term in query_terms)

        if has_term_match or has_phrase_match:
            candidate_documents.append(document)

    # Fallback to all documents if specific BM25 candidate list is empty
    had_keyword_match = bool(candidate_documents)
    if not candidate_documents:
        if category and category.lower() != "all":
            candidate_documents = [d for d in DOCUMENTS if d["category"].lower() == category.lower()]
        else:
            candidate_documents = list(DOCUMENTS)

    # Compute semantic scores
    all_semantic_scores = semantic_scores(query)
    semantic_map = {
        document["filename"]: all_semantic_scores[idx]
        for idx, document in enumerate(DOCUMENTS)
    }

    max_semantic = max(all_semantic_scores) if all_semantic_scores else 0.0
    if not had_keyword_match and max_semantic < 0.25 and not parsed_ops["exact_phrases"]:
        return []

    # Compute BM25 scores
    bm25_map = {
        document["filename"]: bm25_score(query_terms, document)
        for document in candidate_documents
    }

    max_bm25 = max(bm25_map.values()) if bm25_map and max(bm25_map.values()) > 0 else 1.0

    results = []

    # Score and rank candidates
    for document in candidate_documents:
        filename = document["filename"]
        raw_semantic_score = semantic_map.get(filename, 0.0)

        # Keyword Coverage
        matched_tokens = [term for term in set(query_terms) if term in document["tokens"]]
        matched_terms = len(matched_tokens)

        # Relevance gate: ignore documents with 0 keyword matches and weak semantic similarity
        if matched_terms == 0 and not parsed_ops["exact_phrases"] and raw_semantic_score < 0.25:
            continue

        # BM25 Normalized [0, 1]
        bm25 = bm25_map.get(filename, 0.0)
        bm25_normalized = min(1.0, bm25 / max_bm25)

        # Semantic Normalized [0, 1]
        semantic_normalized = max(0.0, min(1.0, (raw_semantic_score + 1.0) / 2.0))

        # Hybrid Combination
        hybrid_score = (w_bm25 * bm25_normalized) + (w_semantic * semantic_normalized)
        score = hybrid_score * 10.0

        # Keyword Coverage Boost
        matched_tokens = [term for term in set(query_terms) if term in document["tokens"]]
        matched_terms = len(matched_tokens)
        total_terms = max(1, len(set(query_terms)))
        keyword_coverage = matched_terms / total_terms
        keyword_boost = keyword_coverage * 1.5
        score += keyword_boost

        # Title Boost
        title_tokens = set(tokenize(document["title"]))
        title_matches = sum(1 for term in query_terms if term in title_tokens)
        title_boost = title_matches * 2.0
        score += title_boost

        # Exact Query / Phrase Boost
        normalized_content = normalize_text(document["content"])
        normalized_query = normalize_text(query)
        exact_query_boost = 2.0 if normalized_query in normalized_content else 0.0
        score += exact_query_boost

        # Explicit Exact Phrase boost (from quotes "...")
        exact_phrase_boost = 0.0
        if parsed_ops["exact_phrases"]:
            for phrase in parsed_ops["exact_phrases"]:
                if normalize_text(phrase) in normalized_content:
                    exact_phrase_boost += 3.5
        score += exact_phrase_boost

        # Bigram Phrase Boost
        phrase_boost = 0.0
        if len(query_terms) >= 2:
            for i in range(len(query_terms) - 1):
                phrase = f"{query_terms[i]} {query_terms[i + 1]}"
                if phrase in normalized_content:
                    phrase_boost += 1.0
        score += phrase_boost

        # Multiple Term Boost
        multiple_term_boost = 1.5 if (len(set(query_terms)) >= 2 and matched_terms >= 2) else 0.0
        score += multiple_term_boost

        # Word Frequency Boost
        word_freq_boost = 0.0
        for term in query_terms:
            cnt = document["tokens"].count(term)
            if cnt > 0:
                word_freq_boost += min(cnt * 0.1, 1.0)
        score += word_freq_boost

        # Typo Penalty
        if corrections:
            score *= (1.0 - 0.25)

        # Quality Multiplier
        quality_multiplier = document["quality_score"] / 100.0
        score *= (0.7 + 0.3 * quality_multiplier)

        # Spam Penalty
        spam_penalty = document["spam_score"] / 100.0
        score *= (1.0 - 0.40 * spam_penalty)

        # Duplicate Penalty
        duplicate_penalty = 0.50 if document["is_duplicate"] else 0.0
        if duplicate_penalty > 0:
            score *= (1.0 - duplicate_penalty)

        # Low Quality Penalty
        low_quality_penalty = 0.30 if document["is_low_quality"] else 0.0
        if low_quality_penalty > 0:
            score *= (1.0 - low_quality_penalty)

        score = max(0.0, score)

        # Ranking Factors Explanation
        factors = []
        if bm25 > 0:
            factors.append("BM25 relevance")
        if semantic_normalized > 0:
            factors.append("Semantic similarity")
        if keyword_coverage > 0:
            factors.append("Keyword coverage")
        if title_boost > 0:
            factors.append("Title match")
        if exact_phrase_boost > 0 or phrase_boost > 0:
            factors.append("Exact phrase match")
        if document["quality_score"] >= 80:
            factors.append("High quality")
        if document["spam_score"] > 0:
            factors.append("Spam penalty")
        if document["is_duplicate"]:
            factors.append("Duplicate penalty")
        if document["is_low_quality"]:
            factors.append("Low-quality penalty")

        # Generate contextual snippet
        snippet = extract_snippet(document["content"], query_terms)

        results.append({
            "title": document["title"],
            "score": round(score, 2),
            "snippet": snippet,
            "content": document["content"],
            "filename": document["filename"],
            "source": document["filename"],
            "category": document["category"],
            "matched_terms": matched_tokens,
            "quality_score": document["quality_score"],
            "spam_score": document["spam_score"],
            "is_low_quality": document["is_low_quality"],
            "is_duplicate": document["is_duplicate"],
            "duplicate_of": document["duplicate_of"],
            "quality_reasons": document["quality_reasons"],
            "corrections": corrections,
            "did_you_mean": did_you_mean,
            "modified_time": document["modified_time"],
            "ranking": {
                "bm25": round(bm25, 4),
                "bm25_normalized": round(bm25_normalized, 4),
                "semantic_score": round(raw_semantic_score, 4),
                "semantic_normalized": round(semantic_normalized, 4),
                "bm25_weight": round(w_bm25, 2),
                "semantic_weight": round(w_semantic, 2),
                "hybrid_score": round(hybrid_score, 4),
                "keyword_coverage": round(keyword_coverage, 2),
                "title_boost": round(title_boost, 2),
                "phrase_boost": round(phrase_boost + exact_phrase_boost, 2),
                "quality_multiplier": round(quality_multiplier, 2),
                "spam_penalty": round(spam_penalty, 2),
                "duplicate_penalty": round(duplicate_penalty, 2),
                "factors": factors
            }
        })

    # Sort results
    if sort_by == "newest":
        results.sort(key=lambda item: item.get("modified_time", 0), reverse=True)
    elif sort_by == "alphabetical":
        results.sort(key=lambda item: item["title"].lower())
    else:  # relevance
        results.sort(key=lambda item: item["score"], reverse=True)

    return results


# =========================================================
# REINDEXING FUNCTIONALITY
# =========================================================

def reindex(force: bool = True) -> Dict[str, Any]:
    """
    Rebuild document database, inverted index, and vector embeddings cache.
    Can be called dynamically via API or CLI.
    """
    global DOCUMENTS, INDEX, DOCUMENT_EMBEDDINGS
    print("\n[REINDEX] Rebuilding SmartSearch knowledge base...")

    DOCUMENTS = build_index()
    INDEX = build_inverted_index(DOCUMENTS)
    DOCUMENT_EMBEDDINGS = load_or_create_embeddings(DOCUMENTS, SEMANTIC_MODEL, force_recompute=force)

    categories = list(set(doc["category"] for doc in DOCUMENTS))
    print(f"[REINDEX] Completed. {len(DOCUMENTS)} documents across {len(categories)} categories.")

    return {
        "status": "success",
        "documents_count": len(DOCUMENTS),
        "categories": categories,
        "model": MODEL_NAME,
        "has_embeddings": DOCUMENT_EMBEDDINGS is not None
    }


if __name__ == "__main__":
    test_q = "python"
    print(f"\nRunning test query: '{test_q}'")
    res = search(test_q)
    print(f"Results returned: {len(res)}")
    for r in res[:3]:
        print(f"- {r['title']} (Score: {r['score']}, Category: {r['category']})")
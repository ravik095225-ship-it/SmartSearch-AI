import os
import re
from typing import List, Dict, Any, Optional

from google import genai
from search_engine import search
from web_search import search_web


# ============================================================
# SMARTSEARCH AI — CONVERSATIONAL RAG & HYBRID RETRIEVAL
# ============================================================

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_RELEVANCE_THRESHOLD = float(os.getenv("LOCAL_RELEVANCE_THRESHOLD", "4.0"))


# ============================================================
# CONVERSATION MEMORY
# ============================================================

CONVERSATION_MEMORY = {
    "last_query": "",
    "last_topic": "",
    "last_keywords": [],
    "history": []
}

LAST_INTERACTION_ID = None


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(
            api_key=api_key
        )
    except Exception:
        return None


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "what", "is", "are", "the", "a", "an",
    "of", "to", "for", "in", "on", "and",
    "or", "how", "why", "where", "when",
    "who", "which", "does", "do", "can",
    "could", "would", "should", "please",
    "tell", "me", "about", "explain",
    "its", "it", "this", "that",
    "these", "those", "they",
    "main", "important", "purpose",
    "features", "feature"
}


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def understand_query(query: str) -> Dict[str, Any]:

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    keywords = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return {
        "original_query": query.strip(),
        "keywords": keywords,
        "word_count": len(words),
        "is_question": query.strip().endswith("?")
    }


# ============================================================
# CONTEXTUAL QUERY
# ============================================================

def is_contextual_query(query: str) -> bool:

    q = query.strip().lower()

    patterns = [
        r"\bits\b",
        r"\bit\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b"
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            q
        ):
            return True

    return False


# ============================================================
# CONTEXT RESOLUTION
# ============================================================

def resolve_context(
    query: str,
    understanding: Dict[str, Any]
) -> str:

    memory = CONVERSATION_MEMORY

    topic = memory.get(
        "last_topic",
        ""
    ).strip()

    if not topic:
        return query

    if not is_contextual_query(query):
        return query

    q = query.strip()

    # --------------------------------------------------------
    # Natural contextual questions
    # --------------------------------------------------------

    patterns = [

        (
            r"^what are its main features\??$",
            f"What are the main features of {topic}?"
        ),

        (
            r"^what are its features\??$",
            f"What are the features of {topic}?"
        ),

        (
            r"^where is it used\??$",
            f"Where is {topic} used?"
        ),

        (
            r"^how is it used\??$",
            f"How is {topic} used?"
        ),

        (
            r"^how does it work\??$",
            f"How does {topic} work?"
        ),

        (
            r"^what is its purpose\??$",
            f"What is the purpose of {topic}?"
        ),

        (
            r"^what is its main purpose\??$",
            f"What is the main purpose of {topic}?"
        ),

        (
            r"^what are its advantages\??$",
            f"What are the advantages of {topic}?"
        ),

        (
            r"^what are its disadvantages\??$",
            f"What are the disadvantages of {topic}?"
        ),

        (
            r"^what are its applications\??$",
            f"What are the applications of {topic}?"
        ),

        (
            r"^what are its benefits\??$",
            f"What are the benefits of {topic}?"
        ),

        (
            r"^what are its types\??$",
            f"What are the types of {topic}?"
        ),

        (
            r"^can you explain it\??$",
            f"Can you explain {topic}?"
        ),

        (
            r"^explain it\??$",
            f"Explain {topic}."
        ),

        (
            r"^tell me more about it\??$",
            f"Tell me more about {topic}."
        )
    ]

    for pattern, replacement in patterns:

        if re.match(
            pattern,
            q,
            flags=re.IGNORECASE
        ):
            return replacement

    # --------------------------------------------------------
    # Generic pronoun replacement
    # --------------------------------------------------------

    resolved = q

    replacements = [
        (r"\bits\b", topic),
        (r"\bit\b", topic),
        (r"\bthis\b", topic),
        (r"\bthat\b", topic),
        (r"\bthey\b", topic),
        (r"\bthese\b", topic),
        (r"\bthose\b", topic)
    ]

    for pattern, replacement in replacements:

        resolved = re.sub(
            pattern,
            topic,
            resolved,
            flags=re.IGNORECASE
        )

    return resolved


# ============================================================
# TOPIC DETECTION
# ============================================================

def detect_topic(
    query: str,
    results: List[Dict[str, Any]]
) -> str:

    """
    Detect the actual topic from the query first.
    This prevents a weak semantic result such as AI
    from replacing Database as the conversation topic.
    """

    q = query.lower()

    topic_aliases = {

        "python": "Python",

        "database": "Database",

        "sql": "Database",

        "networking": "Networking",

        "network": "Networking",

        "tcp": "Networking",

        "ip": "Networking",

        "artificial intelligence": "AI",

        "ai": "AI",

        "machine learning": "AI",

        "neural network": "AI",

        "web development": "Web Development",

        "web": "Web Development",

        "html": "Web Development",

        "css": "Web Development",

        "javascript": "Web Development"
    }

    # Longest match first
    for key in sorted(
        topic_aliases.keys(),
        key=len,
        reverse=True
    ):

        if re.search(
            r"\b" + re.escape(key) + r"\b",
            q
        ):

            return topic_aliases[key]

    # Otherwise use first relevant result
    if results:

        title = results[0].get(
            "title",
            ""
        ).strip()

        if title:
            return title

    return CONVERSATION_MEMORY.get(
        "last_topic",
        ""
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query: str) -> List[str]:

    q = query.lower().strip()

    queries = [q]

    expansion_map = {

        "python": [
            "python programming language",
            "python programming",
            "python features",
            "python uses"
        ],

        "database": [
            "database system",
            "database management",
            "database storage",
            "SQL database"
        ],

        "networking": [
            "computer networking",
            "network protocols",
            "TCP IP",
            "computer communication"
        ],

        "ai": [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "natural language processing"
        ],

        "web development": [
            "web development",
            "web applications",
            "HTML CSS JavaScript",
            "frontend backend"
        ]
    }

    for keyword, variants in expansion_map.items():

        if keyword in q:

            queries.extend(
                variants
            )

    return list(
        dict.fromkeys(
            queries
        )
    )


# ============================================================
# SCORE HELPER
# ============================================================

def get_result_score(
    result: Dict[str, Any]
) -> float:

    ranking = result.get(
        "ranking",
        {}
    )

    return max(

        float(
            result.get(
                "score",
                0
            )
        ),

        float(
            ranking.get(
                "hybrid_score",
                0
            )
        )
    )


# ============================================================
# STRICT RELEVANCE FILTER
# ============================================================

def filter_rag_sources(
    results: List[Dict[str, Any]],
    query: str = ""
) -> List[Dict[str, Any]]:

    if not results:
        return []

    query_lower = query.lower()

    # --------------------------------------------------------
    # Detect exact topic
    # --------------------------------------------------------

    topic = detect_topic(
        query,
        results
    ).lower()

    # --------------------------------------------------------
    # Query keywords
    # --------------------------------------------------------

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query_lower
        )
    )

    meaningful_words = (
        query_words - STOPWORDS
    )

    # --------------------------------------------------------
    # Topic aliases
    # --------------------------------------------------------

    topic_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            topic
        )
    )

    candidates = []

    seen_content = set()

    for result in results:

        # ----------------------------------------------------
        # Duplicate file
        # ----------------------------------------------------

        if result.get(
            "is_duplicate",
            False
        ):
            continue

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        quality = float(
            result.get(
                "quality_score",
                100
            )
        )

        if quality < 35:
            continue

        title = result.get(
            "title",
            ""
        ).strip()

        filename = result.get(
            "filename",
            ""
        ).strip()

        content = result.get(
            "content",
            ""
        ).strip()

        # ----------------------------------------------------
        # Duplicate content
        # ----------------------------------------------------

        fingerprint = re.sub(
            r"\W+",
            " ",
            content.lower()
        ).strip()

        if fingerprint:

            if fingerprint in seen_content:
                continue

            seen_content.add(
                fingerprint
            )

        # ----------------------------------------------------
        # Full text
        # ----------------------------------------------------

        full_text = (
            title +
            " " +
            filename +
            " " +
            content
        ).lower()

        text_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                full_text
            )
        )

        # ----------------------------------------------------
        # Exact topic matching
        # ----------------------------------------------------

        topic_match = False

        if topic:

            topic_match = (
                topic in full_text
                or
                topic.replace(
                    " ",
                    "_"
                ) in full_text
            )

        # ----------------------------------------------------
        # Filename topic matching
        # ----------------------------------------------------

        filename_match = False

        topic_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            topic.lower()
        ).strip("_")

        filename_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            filename.lower()
        ).strip("_")

        if topic_slug and topic_slug in filename_slug:

            filename_match = True

        # ----------------------------------------------------
        # Keyword overlap
        # ----------------------------------------------------

        matched = (
            meaningful_words &
            text_words
        )

        keyword_ratio = 0

        if meaningful_words:

            keyword_ratio = (
                len(matched)
                /
                len(meaningful_words)
            )

        # ----------------------------------------------------
        # Ranking values
        # ----------------------------------------------------

        ranking = result.get(
            "ranking",
            {}
        )

        hybrid = float(
            ranking.get(
                "hybrid_score",
                0
            )
        )

        semantic = float(
            ranking.get(
                "semantic_score",
                0
            )
        )

        coverage = float(
            ranking.get(
                "keyword_coverage",
                0
            )
        )

        # ----------------------------------------------------
        # Build relevance score
        # ----------------------------------------------------

        relevance = 0

        if topic_match:
            relevance += 100

        if filename_match:
            relevance += 80

        relevance += (
            keyword_ratio * 50
        )

        relevance += (
            coverage * 30
        )

        relevance += (
            hybrid * 20
        )

        relevance += (
            semantic * 10
        )

        # ----------------------------------------------------
        # STRICT RULE
        # ----------------------------------------------------

        # If query clearly contains a topic such as Python,
        # don't allow unrelated documents merely because their
        # embeddings are semantically similar.

        if topic:

            if not topic_match and not filename_match:

                # Allow only if there is strong keyword
                # evidence and the query does not represent
                # a direct topic question.
                if keyword_ratio < 0.50:

                    continue

        else:

            if meaningful_words:

                if (
                    keyword_ratio < 0.25
                    and
                    hybrid < 0.65
                    and
                    semantic < 0.65
                ):

                    continue

        candidates.append(
            (
                relevance,
                result
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    filtered = [
        item[1]
        for item in candidates
    ]

    # --------------------------------------------------------
    # Maximum 5 sources
    # --------------------------------------------------------

    return filtered[:5]


# ============================================================
# EXPANDED SEARCH
# ============================================================

def expanded_search(
    query: str
) -> List[Dict[str, Any]]:

    queries = expand_query(
        query
    )

    collected = []

    for expanded_query in queries:

        try:

            results = search(
                expanded_query
            )

            if results:

                collected.extend(
                    results
                )

        except Exception:

            continue

    # --------------------------------------------------------
    # Keep strongest result per file
    # --------------------------------------------------------

    unique = {}

    for result in collected:

        filename = result.get(
            "filename",
            result.get(
                "title",
                "unknown"
            )
        )

        score = get_result_score(
            result
        )

        if filename not in unique:

            unique[filename] = result

        else:

            old_score = get_result_score(
                unique[filename]
            )

            if score > old_score:

                unique[filename] = result

    results = list(
        unique.values()
    )

    # --------------------------------------------------------
    # Strict filtering
    # --------------------------------------------------------

    results = filter_rag_sources(
        results,
        query
    )

    return results


# ============================================================
# RAG RE-RANKING
# ============================================================

def rerank_rag_results(
    results: List[Dict[str, Any]],
    query: str
) -> List[Dict[str, Any]]:

    if not results:
        return []

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    ) - STOPWORDS

    reranked = []

    topic = detect_topic(
        query,
        results
    ).lower()

    for result in results:

        title = result.get(
            "title",
            ""
        ).lower()

        filename = result.get(
            "filename",
            ""
        ).lower()

        content = result.get(
            "content",
            ""
        ).lower()

        text = (
            title +
            " " +
            filename +
            " " +
            content
        )

        text_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text
            )
        )

        overlap = (
            query_words &
            text_words
        )

        keyword_score = 0

        if query_words:

            keyword_score = (
                len(overlap)
                /
                len(query_words)
            )

        ranking = result.get(
            "ranking",
            {}
        )

        hybrid = float(
            ranking.get(
                "hybrid_score",
                0
            )
        )

        semantic = float(
            ranking.get(
                "semantic_score",
                0
            )
        )

        topic_bonus = 0

        if topic and topic in text:

            topic_bonus = 2.0

        final_score = (

            keyword_score * 3.0

            +

            hybrid * 1.5

            +

            semantic * 0.5

            +

            topic_bonus
        )

        result["_rag_score"] = (
            final_score
        )

        reranked.append(
            result
        )

    reranked.sort(
        key=lambda x: x.get(
            "_rag_score",
            0
        ),
        reverse=True
    )

    return reranked


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: List[Dict[str, Any]]
) -> str:

    if not results:
        return "No relevant sources were found."

    parts = []

    for index, result in enumerate(
        results[:5],
        start=1
    ):

        title = result.get(
            "title",
            "Unknown"
        )

        content = result.get(
            "content",
            ""
        ).strip()

        is_web = result.get("is_web", False) or result.get("source_type") == "web"

        if is_web:
            domain = result.get("domain") or result.get("source") or "Web"
            url = result.get("url", "")
            parts.append(
                f"""
SOURCE [{index}] (Web: {domain})

Title:
{title}

URL:
{url}

Content:
{content}
"""
            )
        else:
            filename = result.get(
                "filename",
                "unknown"
            )
            parts.append(
                f"""
SOURCE [{index}] (Local: {filename})

Title:
{title}

File:
{filename}

Content:
{content}
"""
            )

    return "\n".join(
        parts
    )


# ============================================================
# PREPARE SOURCES
# ============================================================

def prepare_sources(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    sources = []

    for index, result in enumerate(
        results[:5],
        start=1
    ):
        is_web = result.get("is_web", False) or result.get("source_type") == "web"

        sources.append({

            "id": index,

            "title": result.get(
                "title",
                "Unknown"
            ),

            "filename": result.get(
                "filename",
                ""
            ),

            "category": result.get(
                "category",
                "Web" if is_web else "General"
            ),

            "snippet": result.get(
                "snippet",
                result.get("content", "")[:180]
            ),

            "content": result.get(
                "content",
                ""
            ),

            "score": result.get(
                "score",
                0
            ),

            "quality_score": result.get(
                "quality_score",
                0
            ),

            "spam_score": result.get(
                "spam_score",
                0
            ),

            "duplicate": result.get(
                "is_duplicate",
                False
            ),

            "url": result.get(
                "url",
                None
            ),

            "domain": result.get(
                "domain",
                None
            ),

            "source_type": "web" if is_web else "local",

            "is_web": is_web
        })

    return sources


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    results: List[Dict[str, Any]],
    ai_status: str = "fallback"
) -> Dict[str, Any]:

    if not results:

        return {
            "score": 0,
            "level": "Low"
        }

    top = results[0]

    if top.get("is_web") or top.get("source_type") == "web":
        score = 80 if ai_status == "gemini" else 70
        level = "High" if score >= 75 else "Medium"
        return {"score": score, "level": level}

    ranking = top.get(
        "ranking",
        {}
    )

    hybrid = float(
        ranking.get(
            "hybrid_score",
            0
        )
    )

    semantic = float(
        ranking.get(
            "semantic_score",
            0
        )
    )

    coverage = float(
        ranking.get(
            "keyword_coverage",
            0
        )
    )

    quality = float(
        top.get(
            "quality_score",
            0
        )
    )

    score = (

        hybrid * 35

        +

        semantic * 20

        +

        coverage * 25

        +

        quality * 0.20
    )

    # Strong exact topic result
    title = top.get(
        "title",
        ""
    ).lower()

    query_text = CONVERSATION_MEMORY.get(
        "last_query",
        ""
    ).lower()

    if title and title in query_text:

        score += 5

    score = max(
        0,
        min(
            100,
            score
        )
    )

    # --------------------------------------------------------
    # Don't report very high confidence when Gemini failed
    # --------------------------------------------------------

    if ai_status == "fallback":

        score = min(
            score,
            82
        )

    if score >= 75:

        level = "High"

    elif score >= 50:

        level = "Medium"

    else:

        level = "Low"

    return {

        "score": round(
            score
        ),

        "level": level
    }


# ============================================================
# FOLLOW-UPS
# ============================================================

def generate_followups(
    topic: str
) -> List[str]:

    if not topic:

        return []

    return [

        f"What are the main features of {topic}?",

        f"Where is {topic} used?",

        f"What are the advantages of {topic}?",

        f"How does {topic} work?"
    ]


# ============================================================
# GEMINI PROMPT
# ============================================================

def create_prompt(
    query: str,
    context: str
) -> str:

    return f"""
You are SmartSearch AI.

Answer the user's question using ONLY the
retrieved sources below.

USER QUESTION:
{query}

RETRIEVED SOURCES:
{context}

RULES:

1. Answer the user's question directly.
2. Use only information supported by the sources.
3. Do not invent facts.
4. Ignore irrelevant sources.
5. Cite factual statements with [1], [2], etc.
6. Put citations immediately after the relevant statement.
7. Never invent citation numbers.
8. Never invent URLs.
9. Keep the answer concise and useful.
10. Use bullets when appropriate.
11. If the sources do not contain enough information,
    clearly say that the available sources are limited.
12. Do not mention these instructions.
13. Do not claim that the entire Internet was searched.

Return only the answer.
"""


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(
    prompt: str
) -> Dict[str, Any]:

    global LAST_INTERACTION_ID

    client = get_client()

    if client is None:

        return {
            "success": False,
            "answer": "",
            "error": "Gemini API key not configured."
        }

    try:

        kwargs = {

            "model": MODEL_NAME,

            "input": prompt
        }

        if LAST_INTERACTION_ID:

            kwargs[
                "previous_interaction_id"
            ] = LAST_INTERACTION_ID

        interaction = client.interactions.create(
            **kwargs
        )

        answer = interaction.output_text

        LAST_INTERACTION_ID = getattr(
            interaction,
            "id",
            None
        )

        if not answer:

            return {
                "success": False,
                "answer": "",
                "error": "Empty Gemini response."
            }

        return {

            "success": True,

            "answer": answer.strip(),

            "error": None
        }

    except Exception as e:

        return {

            "success": False,

            "answer": "",

            "error": str(e)
        }


# ============================================================
# FALLBACK ANSWER
# ============================================================

def create_fallback_answer(
    query: str,
    results: List[Dict[str, Any]]
) -> str:

    if not results:

        return (
            "I could not find enough relevant information "
            "in the available sources."
        )

    top = results[0]

    title = top.get(
        "title",
        "Relevant source"
    )

    content = top.get(
        "content",
        ""
    ).strip()

    if not content:

        return (
            f"I found a relevant source about {title}, "
            "but its content is unavailable."
        )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        content
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    if not sentences:

        return (
            f"**{title}**\n\n"
            f"{content} [1]"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Fallback now uses ONLY the strongest relevant source.
    # This prevents unrelated AI/Database documents appearing.
    # --------------------------------------------------------

    selected = sentences[:5]

    return (
        f"**{title}**\n\n"
        +
        " ".join(selected)
        +
        " [1]"
    )


# ============================================================
# UPDATE MEMORY
# ============================================================

def update_memory(
    query: str,
    answer: str,
    topic: str
):

    CONVERSATION_MEMORY[
        "last_query"
    ] = query

    CONVERSATION_MEMORY[
        "last_topic"
    ] = topic

    understanding = understand_query(
        query
    )

    CONVERSATION_MEMORY[
        "last_keywords"
    ] = understanding[
        "keywords"
    ]

    CONVERSATION_MEMORY[
        "history"
    ].append({

        "query": query,

        "answer": answer,

        "topic": topic
    })

    CONVERSATION_MEMORY[
        "history"
    ] = CONVERSATION_MEMORY[
        "history"
    ][-10:]


# ============================================================
# MAIN GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str
) -> Dict[str, Any]:

    query = query.strip()

    if not query:

        return {

            "answer":
                "Please enter a search question.",

            "query": "",

            "resolved_query": "",

            "sources": [],

            "confidence": {
                "score": 0,
                "level": "Low"
            },

            "followups": [],

            "ai_status": "empty",

            "conversation_topic":
                CONVERSATION_MEMORY[
                    "last_topic"
                ],

            "result_count": 0,

            "error": None
        }

    # --------------------------------------------------------
    # Understand
    # --------------------------------------------------------

    understanding = understand_query(
        query
    )

    # --------------------------------------------------------
    # Resolve context
    # --------------------------------------------------------

    resolved_query = resolve_context(
        query,
        understanding
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = expanded_search(
        resolved_query
    )

    # --------------------------------------------------------
    # Re-rank
    # --------------------------------------------------------

    results = rerank_rag_results(
        results,
        resolved_query
    )

    # --------------------------------------------------------
    # Strict filter AGAIN after reranking
    # --------------------------------------------------------

    results = filter_rag_sources(
        results,
        resolved_query
    )

    # Re-rank once more after filtering
    results = rerank_rag_results(
        results,
        resolved_query
    )

    # --------------------------------------------------------
    # Web search fallback / combine:
    # If local results are empty or their relevance is below LOCAL_RELEVANCE_THRESHOLD
    # --------------------------------------------------------
    max_score = get_result_score(results[0]) if results else 0.0
    if not results or max_score < LOCAL_RELEVANCE_THRESHOLD:
        try:
            web_results = search_web(resolved_query, max_results=6)
            if web_results:
                if not results:
                    results = web_results
                else:
                    results = results + web_results
        except Exception as e:
            print(f"[AI ENGINE] Web search fallback error: {e}")

    # --------------------------------------------------------
    # Determine topic BEFORE updating memory
    # --------------------------------------------------------

    topic = detect_topic(
        resolved_query,
        results
    )

    # --------------------------------------------------------
    # RAG context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    prompt = create_prompt(
        resolved_query,
        context
    )

    gemini_result = generate_with_gemini(
        prompt
    )

    if gemini_result["success"]:

        answer = gemini_result[
            "answer"
        ]

        ai_status = "gemini"

    else:

        answer = create_fallback_answer(
            resolved_query,
            results
        )

        ai_status = "fallback"

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = prepare_sources(
        results
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        results,
        ai_status
    )

    # --------------------------------------------------------
    # Follow-ups
    # --------------------------------------------------------

    followups = generate_followups(
        topic
    )

    # --------------------------------------------------------
    # Save memory
    # --------------------------------------------------------

    update_memory(
        query,
        answer,
        topic
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "answer": answer,

        "query": query,

        "resolved_query":
            resolved_query,

        "sources": sources,

        "confidence": confidence,

        "followups": followups,

        "ai_status": ai_status,

        "conversation_topic": topic,

        "result_count":
            len(results),

        "error":
            gemini_result.get(
                "error"
            )
            if not gemini_result[
                "success"
            ]
            else None
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       SMARTSEARCH AI — PHASE 4")
    print("       Conversational RAG Search")
    print("=" * 60)
    print()

    print("Type 'exit' to stop.")
    print()

    while True:

        try:

            user_query = input(
                "You: "
            ).strip()

        except KeyboardInterrupt:

            print()
            break

        if not user_query:

            continue

        if user_query.lower() in {
            "exit",
            "quit"
        }:

            print(
                "SmartSearch AI stopped."
            )

            break

        result = generate_answer(
            user_query
        )

        print()

        print(
            "Resolved Query:",
            result[
                "resolved_query"
            ]
        )

        print()

        print(
            "AI Status:",
            result[
                "ai_status"
            ]
        )

        print()

        print(
            "Topic:",
            result[
                "conversation_topic"
            ]
        )

        print()

        print(
            "Confidence:",
            result[
                "confidence"
            ]
        )

        print()

        print("Answer:")

        print(
            result[
                "answer"
            ]
        )

        print()

        print("Sources:")

        if result["sources"]:

            for source in result[
                "sources"
            ]:

                print(
                    f"[{source['id']}] "
                    f"{source['title']} "
                    f"— {source['filename']}"
                )

        else:

            print(
                "No relevant sources."
            )

        print()

        print("Follow-ups:")

        for followup in result[
            "followups"
        ]:

            print(
                "-",
                followup
            )

        print()

        if result.get("error"):

            print(
                "Gemini/Error:",
                result["error"]
            )

            print()

        print("-" * 60)
        print()
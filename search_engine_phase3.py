from pathlib import Path
import re
import math
import hashlib
from difflib import get_close_matches

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# SETTINGS
# =========================================================

DOCUMENTS_FOLDER = (
    Path(__file__).parent / "data" / "documents"
)

MODEL_NAME = "all-MiniLM-L6-v2"

BM25_K1 = 1.5
BM25_B = 0.75

BM25_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.40


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "as", "by",
    "this", "that", "are", "was", "be", "from",
    "it", "at", "which", "how", "what", "why",
    "when", "where"
}


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def remove_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def normalize_text(text):

    return " ".join(
        tokenize(text)
    )


# =========================================================
# FINGERPRINT
# =========================================================

def fingerprint(text):

    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# =========================================================
# QUALITY + SPAM ANALYSIS
# =========================================================

def analyze_quality(text):

    tokens = tokenize(text)

    if not tokens:

        return {
            "quality_score": 0,
            "spam_score": 100,
            "is_low_quality": True,
            "reasons": [
                "Empty document"
            ]
        }

    score = 100
    spam_score = 0
    reasons = []

    word_count = len(tokens)

    # Document length
    if word_count < 15:

        score -= 35
        spam_score += 25

        reasons.append(
            "Very short content"
        )

    elif word_count < 30:

        score -= 15
        spam_score += 10

        reasons.append(
            "Short content"
        )

    # Word repetition
    frequencies = {}

    for word in tokens:

        frequencies[word] = (
            frequencies.get(word, 0) + 1
        )

    most_common = max(
        frequencies.values()
    )

    if word_count >= 10:

        repetition_ratio = (
            most_common / word_count
        )

        if repetition_ratio > 0.50:

            score -= 25
            spam_score += 35

            reasons.append(
                "Excessive word repetition"
            )

        elif repetition_ratio > 0.35:

            score -= 10
            spam_score += 15

            reasons.append(
                "High word repetition"
            )

    # Repeated phrases
    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(
            len(tokens) - 1
        )
    ]

    if bigrams:

        phrase_counts = {}

        for phrase in bigrams:

            phrase_counts[phrase] = (
                phrase_counts.get(
                    phrase,
                    0
                ) + 1
            )

        repeated_phrase = max(
            phrase_counts.values()
        )

        if repeated_phrase >= 4:

            score -= 20
            spam_score += 30

            reasons.append(
                "Repeated phrases detected"
            )

    # Promotional symbols
    symbol_count = len(
        re.findall(
            r"[!$%*@#]",
            text
        )
    )

    if symbol_count > 10:

        score -= 15
        spam_score += 20

        reasons.append(
            "Excessive promotional symbols"
        )

    # Uppercase detection
    letters = re.findall(
        r"[a-zA-Z]",
        text
    )

    if len(letters) > 20:

        uppercase = sum(
            1
            for char in letters
            if char.isupper()
        )

        uppercase_ratio = (
            uppercase / len(letters)
        )

        if uppercase_ratio > 0.60:

            score -= 20
            spam_score += 25

            reasons.append(
                "Excessive uppercase text"
            )

    score = max(
        0,
        min(100, score)
    )

    spam_score = max(
        0,
        min(100, spam_score)
    )

    return {
        "quality_score": score,
        "spam_score": spam_score,
        "is_low_quality": score < 50,
        "reasons": reasons
    }


# =========================================================
# BUILD DOCUMENT DATABASE
# =========================================================

def build_index():

    documents = []

    if not DOCUMENTS_FOLDER.exists():

        return documents

    for file_path in DOCUMENTS_FOLDER.glob(
        "*.txt"
    ):

        try:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

        except Exception:

            continue

        if not text:

            continue

        title = (
            file_path.stem
            .replace("_", " ")
            .title()
        )

        tokens = remove_stopwords(
            tokenize(text)
        )

        quality = analyze_quality(
            text
        )

        documents.append({

            "title":
                title,

            "filename":
                file_path.name,

            "content":
                text,

            "tokens":
                tokens,

            "length":
                len(tokens),

            "fingerprint":
                fingerprint(text),

            "quality_score":
                quality[
                    "quality_score"
                ],

            "spam_score":
                quality[
                    "spam_score"
                ],

            "is_low_quality":
                quality[
                    "is_low_quality"
                ],

            "quality_reasons":
                quality[
                    "reasons"
                ],

            "is_duplicate":
                False,

            "duplicate_of":
                None
        })

    # Duplicate detection
    seen = {}

    for document in documents:

        fp = document[
            "fingerprint"
        ]

        if fp in seen:

            original = seen[fp]

            document[
                "is_duplicate"
            ] = True

            document[
                "duplicate_of"
            ] = original[
                "filename"
            ]

            document[
                "spam_score"
            ] = min(
                100,
                document[
                    "spam_score"
                ] + 25
            )

        else:

            seen[fp] = document

    return documents


# =========================================================
# LOAD DOCUMENTS
# =========================================================

print()
print("=" * 60)
print("SMARTSEARCH AI")
print("Loading documents...")
print("=" * 60)

DOCUMENTS = build_index()

print(
    f"Documents loaded: {len(DOCUMENTS)}"
)


# =========================================================
# INVERTED INDEX
# =========================================================

INDEX = {}

for document in DOCUMENTS:

    for word in set(
        document["tokens"]
    ):

        if word not in INDEX:

            INDEX[word] = []

        INDEX[word].append(
            document
        )


# =========================================================
# SEMANTIC MODEL
# =========================================================

print()
print("Loading semantic model...")

try:

    SEMANTIC_MODEL = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Semantic model loaded."
    )

except Exception as error:

    SEMANTIC_MODEL = None

    print(
        "WARNING: Semantic model "
        "could not be loaded."
    )

    print(
        f"Error: {error}"
    )


# =========================================================
# DOCUMENT EMBEDDINGS
# =========================================================

DOCUMENT_EMBEDDINGS = None

if (
    SEMANTIC_MODEL is not None
    and DOCUMENTS
):

    print(
        "Creating document embeddings..."
    )

    document_texts = [
        document["content"]
        for document in DOCUMENTS
    ]

    DOCUMENT_EMBEDDINGS = (
        SEMANTIC_MODEL.encode(
            document_texts,
            show_progress_bar=False
        )
    )

    print(
        "Document embeddings created."
    )


# =========================================================
# BM25
# =========================================================

def document_frequency(term):

    return sum(
        1
        for document in DOCUMENTS
        if term in document["tokens"]
    )


def bm25_idf(term):

    total_documents = len(
        DOCUMENTS
    )

    df = document_frequency(
        term
    )

    if total_documents == 0:

        return 0

    return math.log(
        1
        +
        (
            total_documents
            - df
            + 0.5
        )
        /
        (
            df
            + 0.5
        )
    )


def bm25_score(
    query_terms,
    document
):

    score = 0

    document_length = (
        document["length"]
    )

    average_length = (
        sum(
            doc["length"]
            for doc in DOCUMENTS
        )
        /
        len(DOCUMENTS)
        if DOCUMENTS
        else 1
    )

    frequencies = {}

    for token in document["tokens"]:

        frequencies[token] = (
            frequencies.get(
                token,
                0
            ) + 1
        )

    for term in query_terms:

        if term not in frequencies:

            continue

        tf = frequencies[
            term
        ]

        numerator = (
            tf
            *
            (BM25_K1 + 1)
        )

        denominator = (
            tf
            +
            BM25_K1
            *
            (
                1
                -
                BM25_B
                +
                BM25_B
                *
                (
                    document_length
                    /
                    average_length
                )
            )
        )

        score += (
            bm25_idf(term)
            *
            numerator
            /
            denominator
        )

    return score
# =========================================================
# TYPO CORRECTION
# =========================================================

def correct_typos(query_terms):

    vocabulary = list(
        INDEX.keys()
    )

    corrected = []
    corrections = []

    for term in query_terms:

        if term in INDEX:

            corrected.append(term)
            continue

        matches = get_close_matches(
            term,
            vocabulary,
            n=1,
            cutoff=0.75
        )

        if matches:

            corrected.append(
                matches[0]
            )

            corrections.append(
                f"{term} → {matches[0]}"
            )

        else:

            corrected.append(term)

    return corrected, corrections


# =========================================================
# SEMANTIC SCORES
# =========================================================

def semantic_scores(query):

    if (
        SEMANTIC_MODEL is None
        or DOCUMENT_EMBEDDINGS is None
        or not DOCUMENTS
    ):

        return [0.0] * len(
            DOCUMENTS
        )

    query_embedding = (
        SEMANTIC_MODEL.encode(
            [query]
        )
    )

    similarities = (
        cosine_similarity(
            query_embedding,
            DOCUMENT_EMBEDDINGS
        )[0]
    )

    return [
        float(score)
        for score in similarities
    ]


# =========================================================
# HYBRID SEARCH
# =========================================================

def search(query):

    query = query.strip()

    if not query:
        return []

    # -----------------------------------------------------
    # Query tokens
    # -----------------------------------------------------

    original_terms = (
        remove_stopwords(
            tokenize(query)
        )
    )

    if not original_terms:
        return []

    # -----------------------------------------------------
    # Typo correction
    # -----------------------------------------------------

    query_terms, corrections = (
        correct_typos(
            original_terms
        )
    )

    # -----------------------------------------------------
    # Candidate documents
    # -----------------------------------------------------

    candidate_documents = []

    for document in DOCUMENTS:

        if any(
            term in document["tokens"]
            for term in query_terms
        ):

            candidate_documents.append(
                document
            )

    # -----------------------------------------------------
    # If BM25 finds nothing,
    # use semantic search on all documents
    # -----------------------------------------------------

    if not candidate_documents:

        candidate_documents = list(
            DOCUMENTS
        )

    # -----------------------------------------------------
    # Calculate semantic scores
    # -----------------------------------------------------

    all_semantic_scores = (
        semantic_scores(query)
    )

    semantic_map = {}

    for index, document in enumerate(
        DOCUMENTS
    ):

        semantic_map[
            document["filename"]
        ] = all_semantic_scores[index]

    # -----------------------------------------------------
    # Calculate BM25 scores
    # -----------------------------------------------------

    bm25_map = {}

    for document in candidate_documents:

        bm25_map[
            document["filename"]
        ] = bm25_score(
            query_terms,
            document
        )

    max_bm25 = (
        max(
            bm25_map.values()
        )
        if bm25_map
        else 1
    )

    if max_bm25 <= 0:

        max_bm25 = 1

    results = []

    # =====================================================
    # RANK DOCUMENTS
    # =====================================================

    for document in candidate_documents:

        filename = document[
            "filename"
        ]

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------

        bm25 = bm25_map[
            filename
        ]

        bm25_normalized = (
            bm25 / max_bm25
        )

        # -------------------------------------------------
        # Semantic similarity
        # -------------------------------------------------

        raw_semantic_score = (
            semantic_map[
                filename
            ]
        )

        semantic_normalized = (
            raw_semantic_score + 1
        ) / 2

        semantic_normalized = max(
            0,
            min(
                1,
                semantic_normalized
            )
        )

        # -------------------------------------------------
        # HYBRID SCORE
        # -------------------------------------------------

        hybrid_score = (
            BM25_WEIGHT
            *
            bm25_normalized
        ) + (
            SEMANTIC_WEIGHT
            *
            semantic_normalized
        )

        score = (
            hybrid_score * 10
        )

        # -------------------------------------------------
        # Keyword Coverage
        # -------------------------------------------------

        matched_terms = sum(
            1
            for term in set(
                query_terms
            )
            if term in document[
                "tokens"
            ]
        )

        total_terms = max(
            1,
            len(
                set(query_terms)
            )
        )

        keyword_coverage = (
            matched_terms
            /
            total_terms
        )

        keyword_boost = (
            keyword_coverage * 1.5
        )

        score += keyword_boost

        # -------------------------------------------------
        # Title Boost
        # -------------------------------------------------

        title_tokens = set(
            tokenize(
                document["title"]
            )
        )

        title_matches = sum(
            1
            for term in query_terms
            if term in title_tokens
        )

        title_boost = (
            title_matches * 2
        )

        score += title_boost

        # -------------------------------------------------
        # Exact Query Boost
        # -------------------------------------------------

        normalized_content = (
            normalize_text(
                document["content"]
            )
        )

        normalized_query = (
            normalize_text(query)
        )

        exact_query_boost = 0

        if (
            normalized_query
            in
            normalized_content
        ):

            exact_query_boost = 2

            score += (
                exact_query_boost
            )

        # -------------------------------------------------
        # Phrase Boost
        # -------------------------------------------------

        phrase_boost = 0

        if len(query_terms) >= 2:

            for i in range(
                len(query_terms) - 1
            ):

                phrase = (
                    query_terms[i]
                    + " "
                    + query_terms[i + 1]
                )

                if (
                    phrase
                    in
                    normalized_content
                ):

                    phrase_boost += 1

        score += phrase_boost

        # -------------------------------------------------
        # Multiple Term Boost
        # -------------------------------------------------

        multiple_term_boost = 0

        if (
            len(set(query_terms)) >= 2
            and
            matched_terms >= 2
        ):

            multiple_term_boost = 1.5

            score += (
                multiple_term_boost
            )

        # -------------------------------------------------
        # Word Frequency Boost
        # -------------------------------------------------

        word_frequency_boost = 0

        for term in query_terms:

            count = document[
                "tokens"
            ].count(term)

            if count > 0:

                word_frequency_boost += min(
                    count * 0.1,
                    1
                )

        score += (
            word_frequency_boost
        )

        # -------------------------------------------------
        # Typo Penalty
        # -------------------------------------------------

        typo_penalty = 0

        if corrections:

            typo_penalty = 0.25

            score *= (
                1 - typo_penalty
            )

        # -------------------------------------------------
        # Quality Multiplier
        # -------------------------------------------------

        quality_multiplier = (
            document[
                "quality_score"
            ]
            /
            100
        )

        score *= (
            0.7
            +
            0.3
            *
            quality_multiplier
        )

        # -------------------------------------------------
        # Spam Penalty
        # -------------------------------------------------

        spam_penalty = (
            document[
                "spam_score"
            ]
            /
            100
        )

        score *= (
            1
            -
            0.40
            *
            spam_penalty
        )

        # -------------------------------------------------
        # Duplicate Penalty
        # -------------------------------------------------

        duplicate_penalty = 0

        if document[
            "is_duplicate"
        ]:

            duplicate_penalty = 0.50

            score *= (
                1
                -
                duplicate_penalty
            )

        # -------------------------------------------------
        # Low Quality Penalty
        # -------------------------------------------------

        low_quality_penalty = 0

        if document[
            "is_low_quality"
        ]:

            low_quality_penalty = 0.30

            score *= (
                1
                -
                low_quality_penalty
            )

        score = max(
            0,
            score
        )

        # =================================================
        # RANKING FACTORS
        # =================================================

        factors = []

        if bm25 > 0:

            factors.append(
                "BM25 relevance"
            )

        if semantic_normalized > 0:

            factors.append(
                "Semantic similarity"
            )

        if keyword_coverage > 0:

            factors.append(
                "Keyword coverage"
            )

        if title_boost > 0:

            factors.append(
                "Title match"
            )

        if phrase_boost > 0:

            factors.append(
                "Exact phrase"
            )

        if document[
            "quality_score"
        ] >= 80:

            factors.append(
                "High quality"
            )

        if document[
            "spam_score"
        ] > 0:

            factors.append(
                "Spam penalty"
            )

        if document[
            "is_duplicate"
        ]:

            factors.append(
                "Duplicate penalty"
            )

        if document[
            "is_low_quality"
        ]:

            factors.append(
                "Low-quality penalty"
            )

        # =================================================
        # RESULT
        # =================================================

        results.append({

            "title":
                document["title"],

            "score":
                round(
                    score,
                    2
                ),

            "content":
                document["content"],

            "filename":
                document["filename"],

            "quality_score":
                document[
                    "quality_score"
                ],

            "spam_score":
                document[
                    "spam_score"
                ],

            "is_low_quality":
                document[
                    "is_low_quality"
                ],

            "is_duplicate":
                document[
                    "is_duplicate"
                ],

            "duplicate_of":
                document[
                    "duplicate_of"
                ],

            "quality_reasons":
                document[
                    "quality_reasons"
                ],

            "corrections":
                corrections,

            "ranking": {

                "bm25":
                    round(
                        bm25,
                        4
                    ),

                "bm25_normalized":
                    round(
                        bm25_normalized,
                        4
                    ),

                "semantic_score":
                    round(
                        raw_semantic_score,
                        4
                    ),

                "semantic_normalized":
                    round(
                        semantic_normalized,
                        4
                    ),

                "bm25_weight":
                    BM25_WEIGHT,

                "semantic_weight":
                    SEMANTIC_WEIGHT,

                "hybrid_score":
                    round(
                        hybrid_score,
                        4
                    ),

                "keyword_coverage":
                    round(
                        keyword_coverage,
                        2
                    ),

                "title_boost":
                    round(
                        title_boost,
                        2
                    ),

                "phrase_boost":
                    round(
                        phrase_boost,
                        2
                    ),

                "quality_multiplier":
                    round(
                        quality_multiplier,
                        2
                    ),

                "spam_penalty":
                    round(
                        spam_penalty,
                        2
                    ),

                "duplicate_penalty":
                    round(
                        duplicate_penalty,
                        2
                    ),

                "factors":
                    factors
            }
        })

    # =====================================================
    # FINAL SORT
    # =====================================================

    results.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True
    )

    return results
import math
import re

from incident_rag import (
    search_knowledge,
)


# =========================================================
# Configuration
# =========================================================

VECTOR_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25


# =========================================================
# Normalize text into searchable terms
# =========================================================

def tokenize(text):
    """
    Convert text into normalized keyword tokens.

    Example:

    "Column amount cannot be resolved"

    becomes:

    {
        "column",
        "amount",
        "cannot",
        "resolved"
    }
    """

    if not text:
        return set()

    tokens = re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower(),
    )

    # -----------------------------------------------------
    # Very small stop-word list.
    #
    # We intentionally keep this simple.
    # -----------------------------------------------------

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "is",
        "was",
        "were",
        "be",
        "with",
        "by",
        "from",
        "this",
        "that",
    }

    return {
        token
        for token in tokens
        if (
            token not in stop_words
            and len(token) > 1
        )
    }


# =========================================================
# Keyword overlap score
# =========================================================

def keyword_score(
    query,
    document_text,
):
    """
    Calculate a simple lexical overlap score.

    Score range:
        0.0 -> no overlap
        1.0 -> all query terms matched

    This is intentionally lightweight.

    It gives us the core idea behind lexical/BM25-style
    retrieval without adding Elasticsearch/OpenSearch.
    """

    query_tokens = tokenize(
        query
    )

    document_tokens = tokenize(
        document_text
    )

    if not query_tokens:
        return 0.0

    matched = (
        query_tokens
        & document_tokens
    )

    return (
        len(matched)
        / len(query_tokens)
    )


# =========================================================
# Optional phrase bonus
# =========================================================

def phrase_bonus(
    query,
    document_text,
):
    """
    Reward exact phrase matches.

    Example:

        query:
        "partial write"

        document:
        "failure occurred after partial write"

    This gets a small bonus.
    """

    query = (
        query
        .strip()
        .lower()
    )

    document_text = (
        document_text
        .lower()
    )

    if (
        query
        and query in document_text
    ):
        return 0.10

    return 0.0


# =========================================================
# Metadata bonus
# =========================================================

def metadata_bonus(
    result,
    platform=None,
    environment=None,
    incident_type=None,
):
    """
    Add a small ranking bonus when metadata strongly
    matches the query context.

    Hard filtering still happens in search_knowledge().
    This is only an additional soft ranking signal.
    """

    metadata = result.get(
        "metadata",
        {},
    ) or {}

    bonus = 0.0

    if (
        platform
        and metadata.get(
            "platform"
        )
        == platform
    ):
        bonus += 0.03

    if (
        environment
        and metadata.get(
            "environment"
        )
        == environment
    ):
        bonus += 0.03

    if (
        incident_type
        and metadata.get(
            "incident_type"
        )
        == incident_type
    ):
        bonus += 0.04

    return bonus


# =========================================================
# Hybrid score
# =========================================================

def calculate_hybrid_score(
    query,
    result,
    platform=None,
    environment=None,
    incident_type=None,
):
    """
    Combine:

    1. Vector similarity
    2. Keyword overlap
    3. Phrase bonus
    4. Metadata bonus

    into one hybrid ranking score.
    """

    vector_similarity = float(
        result.get(
            "similarity",
            0.0,
        )
        or 0.0
    )

    document_text = (
        result.get(
            "chunk_text",
            ""
        )
    )

    lexical_score = (
        keyword_score(
            query,
            document_text,
        )
    )

    exact_phrase_bonus = (
        phrase_bonus(
            query,
            document_text,
        )
    )

    context_bonus = (
        metadata_bonus(
            result=result,
            platform=platform,
            environment=environment,
            incident_type=incident_type,
        )
    )

    score = (
        VECTOR_WEIGHT
        * vector_similarity
        +
        KEYWORD_WEIGHT
        * lexical_score
        +
        exact_phrase_bonus
        +
        context_bonus
    )

    return {
        "hybrid_score":
            score,

        "vector_score":
            vector_similarity,

        "keyword_score":
            lexical_score,

        "phrase_bonus":
            exact_phrase_bonus,

        "metadata_bonus":
            context_bonus,
    }


# =========================================================
# Hybrid retrieval
# =========================================================

def hybrid_search(
    query,
    top_k=3,
    candidate_k=10,
    platform=None,
    environment=None,
    document_type=None,
    incident_type=None,
    min_similarity=None,
):
    """
    Retrieve more candidates using vector search,
    then rerank them using hybrid scoring.

    Example:

        candidate_k = 10
        top_k = 3

    Vector DB retrieves 10 candidates.
    Hybrid reranker chooses the best 3.
    """

    # -----------------------------------------------------
    # Stage 1:
    # Candidate generation using existing pgvector search.
    # -----------------------------------------------------

    candidates = search_knowledge(
        query=query,
        top_k=candidate_k,
        platform=platform,
        environment=environment,
        document_type=document_type,
        incident_type=incident_type,
        min_similarity=min_similarity,
    )

    reranked = []

    # -----------------------------------------------------
    # Stage 2:
    # Hybrid reranking.
    # -----------------------------------------------------

    for candidate in candidates:

        scoring = calculate_hybrid_score(
            query=query,
            result=candidate,
            platform=platform,
            environment=environment,
            incident_type=incident_type,
        )

        enriched = dict(
            candidate
        )

        enriched.update(
            scoring
        )

        reranked.append(
            enriched
        )

    # -----------------------------------------------------
    # Highest hybrid score first
    # -----------------------------------------------------

    reranked.sort(
        key=lambda item: item[
            "hybrid_score"
        ],
        reverse=True,
    )

    return reranked[
        :top_k
    ]


# =========================================================
# Simple query expansion
# =========================================================

def expand_incident_query(
    query,
):
    """
    Lightweight deterministic query expansion.

    This is intentionally small.

    In a larger system an LLM could rewrite the query,
    but for common production errors we can enrich the
    query cheaply and deterministically.
    """

    normalized = (
        query.lower()
    )

    expansions = []

    if (
        "column"
        in normalized
        and (
            "not resolved"
            in normalized
            or "cannot be resolved"
            in normalized
        )
    ):
        expansions.extend(
            [
                "schema change",
                "column rename",
                "schema drift",
            ]
        )

    if (
        "duplicate"
        in normalized
        or "duplicated"
        in normalized
    ):
        expansions.extend(
            [
                "partial write",
                "retry",
                "idempotency",
            ]
        )

    if (
        "slow"
        in normalized
        or "runtime"
        in normalized
        or "latency"
        in normalized
    ):
        expansions.extend(
            [
                "performance degradation",
                "runtime anomaly",
                "freshness delay",
            ]
        )

    if not expansions:
        return query

    expanded_query = (
        query
        + " "
        + " ".join(
            expansions
        )
    )

    return expanded_query


# =========================================================
# Enhanced RAG search
# =========================================================

def enhanced_search(
    query,
    top_k=3,
    candidate_k=10,
    platform=None,
    environment=None,
    document_type=None,
    incident_type=None,
    min_similarity=None,
):
    """
    Full enhanced RAG retrieval flow:

        Original query
            ↓
        Query expansion
            ↓
        pgvector candidate retrieval
            ↓
        Hybrid reranking
            ↓
        Top-K results
    """

    expanded_query = (
        expand_incident_query(
            query
        )
    )

    results = hybrid_search(
        query=
            expanded_query,

        top_k=
            top_k,

        candidate_k=
            candidate_k,

        platform=
            platform,

        environment=
            environment,

        document_type=
            document_type,

        incident_type=
            incident_type,

        min_similarity=
            min_similarity,
    )

    return {
        "original_query":
            query,

        "expanded_query":
            expanded_query,

        "results":
            results,
    }


# =========================================================
# Pretty-print results
# =========================================================

def print_results(
    response,
):

    print(
        "\n========== ENHANCED RAG =========="
    )

    print(
        "\nOriginal Query:"
    )

    print(
        response[
            "original_query"
        ]
    )

    print(
        "\nExpanded Query:"
    )

    print(
        response[
            "expanded_query"
        ]
    )

    print(
        "\nResults:"
    )

    for index, result in enumerate(
        response[
            "results"
        ],
        start=1,
    ):

        print(
            "\n----------------------------------------"
        )

        print(
            f"Result #{index}"
        )

        print(
            "Document:",
            result.get(
                "document_id"
            ),
        )

        print(
            "Type:",
            result.get(
                "document_type"
            ),
        )

        print(
            "Title:",
            result.get(
                "title"
            ),
        )

        print(
            "Vector Score:",
            round(
                result.get(
                    "vector_score",
                    0.0,
                ),
                4,
            ),
        )

        print(
            "Keyword Score:",
            round(
                result.get(
                    "keyword_score",
                    0.0,
                ),
                4,
            ),
        )

        print(
            "Hybrid Score:",
            round(
                result.get(
                    "hybrid_score",
                    0.0,
                ),
                4,
            ),
        )

        print(
            "\nChunk:"
        )

        print(
            result.get(
                "chunk_text"
            )
        )


# =========================================================
# Demo
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Test 1:
    # Schema problem
    # -----------------------------------------------------

    query = (
        "Silver transform failed because "
        "column amount cannot be resolved"
    )

    response = enhanced_search(
        query=query,
        top_k=3,
        candidate_k=10,
        platform="databricks",
        environment="production",
    )

    print_results(
        response
    )
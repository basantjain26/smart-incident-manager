import os

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector


load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


def get_connection():
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

    register_vector(conn)

    return conn


def create_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def search_knowledge(
    query,
    top_k=3,
    platform=None,
    environment=None,
    document_type=None,
    incident_type=None,
    min_similarity=None,
):
    """
    Search historical incidents and runbooks using:
    1. Metadata filtering
    2. Vector similarity
    3. Top-K retrieval
    4. Optional similarity threshold
    """

    query_embedding = create_embedding(query)

    conditions = []
    filter_params = []

    if platform:
        conditions.append(
            "metadata->>'platform' = %s"
        )
        filter_params.append(platform)

    if environment:
        conditions.append(
            "metadata->>'environment' = %s"
        )
        filter_params.append(environment)

    if document_type:
        conditions.append(
            "document_type = %s"
        )
        filter_params.append(document_type)

    if incident_type:
        conditions.append(
            "metadata->>'incident_type' = %s"
        )
        filter_params.append(incident_type)

    where_clause = ""

    if conditions:
        where_clause = (
            "WHERE " + " AND ".join(conditions)
        )

    sql = f"""
        SELECT
            document_id,
            document_type,
            title,
            chunk_index,
            chunk_text,
            metadata,
            1 - (embedding <=> %s) AS similarity
        FROM knowledge_chunks
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    params = [
        Vector(query_embedding),
        *filter_params,
        Vector(query_embedding),
        top_k,
    ]

    with get_connection() as conn:
        rows = conn.execute(
            sql,
            params,
        ).fetchall()

    results = []

    for row in rows:
        similarity = float(row[6])

        if (
            min_similarity is not None
            and similarity < min_similarity
        ):
            continue

        results.append(
            {
                "document_id": row[0],
                "document_type": row[1],
                "title": row[2],
                "chunk_index": row[3],
                "chunk_text": row[4],
                "metadata": row[5],
                "similarity": round(
                    similarity,
                    4,
                ),
            }
        )

    return results


def print_results(results):
    if not results:
        print("\nNo relevant knowledge found.")
        return

    for result in results:
        print("\n--------------------------------")
        print(
            "Document:",
            result["document_id"],
        )
        print(
            "Type:",
            result["document_type"],
        )
        print(
            "Title:",
            result["title"],
        )
        print(
            "Chunk:",
            result["chunk_index"],
        )
        print(
            "Metadata:",
            result["metadata"],
        )
        print(
            "Similarity:",
            result["similarity"],
        )
        print("\nText:")
        print(
            result["chunk_text"]
        )


if __name__ == "__main__":

    query = """
    Spark Silver pipeline failed because the column
    amount cannot be resolved. Schema comparison
    shows amount was removed and
    transaction_amount was added.
    """

    results = search_knowledge(
        query=query,
        top_k=5,
        platform="databricks",
        environment="production",
        min_similarity=0.60,
    )

    print_results(results)
import json
import os
import re

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector


load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def create_table(conn):

    conn.execute(
        "CREATE EXTENSION IF NOT EXISTS vector"
    )

    conn.commit()

    register_vector(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            document_type TEXT NOT NULL,
            title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(1536) NOT NULL,
            UNIQUE(document_id, chunk_index)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        knowledge_chunks_embedding_idx
        ON knowledge_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )

    conn.commit()


def chunk_document(content):

    sections = re.split(
        r"(?=^## )",
        content,
        flags=re.MULTILINE
    )

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]


def create_embedding(text):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def load_documents():

    documents = []

    for path in [
        "data/historical_incidents.json",
        "data/runbooks.json"
    ]:
        with open(path) as f:
            documents.extend(json.load(f))

    return documents


def index_document(conn, document):

    chunks = chunk_document(
        document["content"]
    )

    # Remove old chunks when re-indexing this document.
    conn.execute(
        """
        DELETE FROM knowledge_chunks
        WHERE document_id = %s
        """,
        (document["id"],)
    )

    for chunk_index, chunk_text in enumerate(chunks):

        embedding = create_embedding(chunk_text)

        conn.execute(
            """
            INSERT INTO knowledge_chunks (
                document_id,
                document_type,
                title,
                chunk_index,
                chunk_text,
                metadata,
                embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s::jsonb, %s
            )
            """,
            (
                document["id"],
                document["type"],
                document["title"],
                chunk_index,
                chunk_text,
                json.dumps(document["metadata"]),
                Vector(embedding),
            )
        )

        print(
            f"Indexed {document['id']} "
            f"chunk {chunk_index}"
        )

    conn.commit()


def main():

    with get_connection() as conn:

        create_table(conn)

        documents = load_documents()

        for document in documents:
            index_document(conn, document)

    print("\nKnowledge indexing complete.")


if __name__ == "__main__":
    main()
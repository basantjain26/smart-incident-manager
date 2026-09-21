import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector


# =========================================================
# Configuration
# =========================================================

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

EMBEDDING_DIMENSION = 1536


DB_HOST = os.getenv(
    "DB_HOST",
    "127.0.0.1",
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "5435",
    )
)

DB_NAME = os.getenv(
    "DB_NAME",
    "incident_manager",
)

DB_USER = os.getenv(
    "DB_USER",
    "incident_user",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    "incident_pass",
)


# =========================================================
# Knowledge files
# =========================================================

HISTORICAL_INCIDENTS_FILE = Path(
    "data/historical_incidents.json"
)

RUNBOOKS_FILE = Path(
    "data/runbooks.json"
)


# =========================================================
# PostgreSQL connection
# =========================================================

def get_connection():

    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    register_vector(conn)

    return conn


# =========================================================
# Database table
# =========================================================

def create_knowledge_table():

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,

                    document_id TEXT NOT NULL,

                    document_type TEXT NOT NULL,

                    title TEXT NOT NULL,

                    chunk_index INTEGER NOT NULL,

                    chunk_text TEXT NOT NULL,

                    metadata JSONB,

                    embedding VECTOR(1536),

                    UNIQUE(
                        document_id,
                        chunk_index
                    )
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    knowledge_chunks_embedding_hnsw_idx
                ON knowledge_chunks
                USING hnsw (
                    embedding vector_cosine_ops
                );
                """
            )

        conn.commit()


# =========================================================
# Embedding generation
# =========================================================

def create_embedding(text):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[
        0
    ].embedding


# =========================================================
# Utility JSON loader
# =========================================================

def load_json_file(path):

    if not path.exists():
        return []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =========================================================
# Utility JSON writer
# =========================================================

def write_json_file(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )


# =========================================================
# Normalize arbitrary values for text
# =========================================================

def value_to_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        list,
    ):

        return "\n".join(
            value_to_text(item)
            for item in value
        )

    if isinstance(
        value,
        dict,
    ):

        return json.dumps(
            value,
            indent=2,
            default=str,
        )

    return str(value)


# =========================================================
# Build markdown-like incident document
# =========================================================

def build_incident_document(
    incident,
):

    incident_id = incident.get(
        "incident_id",
        "UNKNOWN",
    )

    pipeline = incident.get(
        "pipeline",
        "unknown",
    )

    title = (
        incident.get("title")
        or f"Incident {incident_id} - {pipeline}"
    )

    symptoms = (
        incident.get("symptoms")
        or incident.get("description")
        or incident.get(
            "original_status",
            "",
        )
    )

    investigation = (
        incident.get("investigation")
        or incident.get(
            "current_evidence",
            "",
        )
    )

    root_cause = incident.get(
        "root_cause",
        "",
    )

    resolution = (
        incident.get("resolution")
        or incident.get(
            "remediation",
            "",
        )
    )

    validation = incident.get(
        "validation",
        "",
    )

    lessons = incident.get(
        "lessons_learned",
        "",
    )

    document = f"""
## Incident

Incident ID: {incident_id}

Pipeline: {pipeline}

Task: {incident.get("task", "")}

Status: {incident.get("final_status", incident.get("status", ""))}

## Symptoms

{value_to_text(symptoms)}

## Investigation

{value_to_text(investigation)}

## Root Cause

{value_to_text(root_cause)}

## Resolution

{value_to_text(resolution)}

## Validation

{value_to_text(validation)}

## Lessons Learned

{value_to_text(lessons)}
""".strip()

    return title, document


# =========================================================
# Build markdown-like runbook document
# =========================================================

def build_runbook_document(
    runbook,
):

    runbook_id = runbook.get(
        "runbook_id",
        "UNKNOWN",
    )

    title = runbook.get(
        "title",
        f"Runbook {runbook_id}",
    )

    symptoms = runbook.get(
        "symptoms",
        "",
    )

    diagnosis = (
        runbook.get("diagnosis")
        or runbook.get(
            "diagnostic_steps",
            "",
        )
    )

    recovery = (
        runbook.get("recovery")
        or runbook.get(
            "procedure",
            "",
        )
    )

    validation = (
        runbook.get("validation")
        or runbook.get(
            "validation_steps",
            "",
        )
    )

    prevention = (
        runbook.get("prevention")
        or runbook.get(
            "prevention_steps",
            "",
        )
    )

    document = f"""
## Symptoms

{value_to_text(symptoms)}

## Diagnosis

{value_to_text(diagnosis)}

## Recovery

{value_to_text(recovery)}

## Validation

{value_to_text(validation)}

## Prevention

{value_to_text(prevention)}
""".strip()

    return title, document


# =========================================================
# Structure-aware chunking
# =========================================================

def chunk_document(
    document_text,
):

    chunks = []

    current_heading = None
    current_lines = []

    for line in document_text.splitlines():

        line = line.rstrip()

        if line.startswith(
            "## "
        ):

            if (
                current_heading
                or current_lines
            ):

                chunk = "\n".join(
                    [
                        current_heading
                        or "",
                        *current_lines,
                    ]
                ).strip()

                if chunk:
                    chunks.append(
                        chunk
                    )

            current_heading = line
            current_lines = []

        else:
            current_lines.append(
                line
            )

    if (
        current_heading
        or current_lines
    ):

        chunk = "\n".join(
            [
                current_heading
                or "",
                *current_lines,
            ]
        ).strip()

        if chunk:
            chunks.append(
                chunk
            )

    return chunks


# =========================================================
# Normalize historical incident metadata
# =========================================================

def incident_metadata(
    incident,
):

    return {
        "incident_type":
            incident.get(
                "incident_type",
                "unknown",
            ),

        "platform":
            incident.get(
                "platform",
                "databricks",
            ),

        "environment":
            incident.get(
                "environment",
                "production",
            ),

        "severity":
            incident.get(
                "severity",
            ),

        "pipeline":
            incident.get(
                "pipeline",
            ),

        "final_status":
            incident.get(
                "final_status",
                incident.get(
                    "status"
                ),
            ),
    }


# =========================================================
# Normalize runbook metadata
# =========================================================

def runbook_metadata(
    runbook,
):

    return {
        "incident_type":
            runbook.get(
                "incident_type",
                "unknown",
            ),

        "platform":
            runbook.get(
                "platform",
                "databricks",
            ),

        "environment":
            runbook.get(
                "environment",
                "production",
            ),

        "version":
            runbook.get(
                "version",
                "1.0",
            ),
    }


# =========================================================
# Index one document
# =========================================================

def index_document(
    document_id,
    document_type,
    title,
    document_text,
    metadata,
):

    chunks = chunk_document(
        document_text
    )

    with get_connection() as conn:

        with conn.cursor() as cur:

            # ---------------------------------------------
            # Delete previous chunks for same document
            #
            # This makes re-indexing idempotent.
            # ---------------------------------------------

            cur.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE document_id = %s
                AND document_type = %s
                """,
                (
                    document_id,
                    document_type,
                ),
            )

            # ---------------------------------------------
            # Insert refreshed chunks
            # ---------------------------------------------

            for chunk_index, chunk_text in enumerate(
                chunks
            ):

                embedding = create_embedding(
                    chunk_text
                )

                cur.execute(
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
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        document_id,
                        document_type,
                        title,
                        chunk_index,
                        chunk_text,
                        json.dumps(
                            metadata
                        ),
                        embedding,
                    ),
                )

        conn.commit()

    return len(chunks)


# =========================================================
# Index historical incidents
# =========================================================

def index_historical_incidents():

    incidents = load_json_file(
        HISTORICAL_INCIDENTS_FILE
    )

    indexed_documents = 0
    indexed_chunks = 0

    for incident in incidents:

        document_id = incident.get(
            "incident_id"
        )

        if not document_id:
            continue

        title, document_text = (
            build_incident_document(
                incident
            )
        )

        chunk_count = index_document(
            document_id=document_id,
            document_type="historical_incident",
            title=title,
            document_text=document_text,
            metadata=incident_metadata(
                incident
            ),
        )

        indexed_documents += 1
        indexed_chunks += chunk_count

    return {
        "documents":
            indexed_documents,

        "chunks":
            indexed_chunks,
    }


# =========================================================
# Index runbooks
# =========================================================

def index_runbooks():

    runbooks = load_json_file(
        RUNBOOKS_FILE
    )

    indexed_documents = 0
    indexed_chunks = 0

    for runbook in runbooks:

        document_id = runbook.get(
            "runbook_id"
        )

        if not document_id:
            continue

        title, document_text = (
            build_runbook_document(
                runbook
            )
        )

        chunk_count = index_document(
            document_id=document_id,
            document_type="runbook",
            title=title,
            document_text=document_text,
            metadata=runbook_metadata(
                runbook
            ),
        )

        indexed_documents += 1
        indexed_chunks += chunk_count

    return {
        "documents":
            indexed_documents,

        "chunks":
            indexed_chunks,
    }


# =========================================================
# Full index
# =========================================================

def index_knowledge():

    create_knowledge_table()

    print(
        "\nIndexing historical incidents..."
    )

    incident_result = (
        index_historical_incidents()
    )

    print(
        json.dumps(
            incident_result,
            indent=2,
        )
    )

    print(
        "\nIndexing runbooks..."
    )

    runbook_result = (
        index_runbooks()
    )

    print(
        json.dumps(
            runbook_result,
            indent=2,
        )
    )

    return {
        "historical_incidents":
            incident_result,

        "runbooks":
            runbook_result,
    }


# =========================================================
# Persist resolved incident
# =========================================================

def save_resolved_incident(
    incident,
    root_cause,
    remediation,
    execution_result,
    validation_result,
):

    incidents = load_json_file(
        HISTORICAL_INCIDENTS_FILE
    )

    incident_id = incident[
        "incident_id"
    ]

    resolved_record = {
        "incident_id":
            incident_id,

        "title":
            (
                f"Resolved incident "
                f"{incident_id} - "
                f"{incident.get('pipeline')}"
            ),

        "incident_type":
            incident.get(
                "incident_type",
                "duplicate_data",
            ),

        "platform":
            incident.get(
                "platform",
                "databricks",
            ),

        "environment":
            incident.get(
                "environment",
                "production",
            ),

        "severity":
            incident.get(
                "severity",
                "high",
            ),

        "pipeline":
            incident.get(
                "pipeline"
            ),

        "task":
            incident.get(
                "task"
            ),

        "source":
            incident.get(
                "source"
            ),

        "run_id":
            incident.get(
                "run_id"
            ),

        "original_status":
            incident.get(
                "status"
            ),

        "root_cause":
            root_cause,

        "remediation":
            remediation,

        "execution_result":
            execution_result,

        "validation":
            validation_result,

        "final_status":
            "RESOLVED",

        "resolved_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "lessons_learned":
            (
                "The remediation was human approved, "
                "executed successfully, and verified "
                "through deterministic post-remediation "
                "validation before the incident was added "
                "to trusted historical knowledge."
            ),
    }

    # -----------------------------------------------------
    # Idempotency:
    # remove older copy of same incident before append
    # -----------------------------------------------------

    incidents = [
        existing
        for existing in incidents
        if existing.get(
            "incident_id"
        ) != incident_id
    ]

    incidents.append(
        resolved_record
    )

    write_json_file(
        HISTORICAL_INCIDENTS_FILE,
        incidents,
    )

    return resolved_record


# =========================================================
# Update/create runbook only when new learning exists
# =========================================================

def update_runbook_if_needed(
    incident,
    remediation,
    validation_result,
    new_learning=False,
):

    if not new_learning:

        return {
            "updated": False,
            "reason":
                (
                    "No new procedural learning "
                    "was identified."
                ),
        }

    runbooks = load_json_file(
        RUNBOOKS_FILE
    )

    runbook_id = (
        f"RB-{incident['incident_id']}"
    )

    new_runbook = {
        "runbook_id":
            runbook_id,

        "title":
            (
                "Recovery procedure learned from "
                f"{incident['incident_id']}"
            ),

        "incident_type":
            incident.get(
                "incident_type",
                "resolved_incident_learning",
            ),

        "platform":
            incident.get(
                "platform",
                "databricks",
            ),

        "environment":
            incident.get(
                "environment",
                "production",
            ),

        "version":
            "1.0",

        "symptoms": [
            (
                f"Production incident affecting "
                f"{incident.get('pipeline')}."
            )
        ],

        "diagnosis": [
            "Gather current production evidence.",
            "Review job logs and operational metrics.",
            "Search validated historical incidents.",
            "Confirm root cause before remediation.",
        ],

        "procedure": [
            (
                "Use remediation action: "
                f"{remediation.get('action')}."
            ),
            (
                "Perform a dry-run before "
                "production execution."
            ),
            (
                "Require human approval before "
                "executing the action."
            ),
        ],

        "validation": [
            (
                "Confirm the recovery run "
                "completed successfully."
            ),
            (
                "Verify expected record counts "
                "and duplicate tolerances."
            ),
            (
                "Verify freshness and other "
                "pipeline health conditions."
            ),
        ],

        "prevention": [
            (
                "Use the validated incident "
                "as future operational evidence."
            )
        ],

        "source_incident":
            incident[
                "incident_id"
            ],

        "validation_required":
            validation_result.get(
                "validation_passed",
                False,
            ),
    }

    # -----------------------------------------------------
    # Idempotent update
    # -----------------------------------------------------

    runbooks = [
        existing
        for existing in runbooks
        if existing.get(
            "runbook_id"
        ) != runbook_id
    ]

    runbooks.append(
        new_runbook
    )

    write_json_file(
        RUNBOOKS_FILE,
        runbooks,
    )

    return {
        "updated":
            True,

        "runbook":
            new_runbook,
    }


# =========================================================
# Closed-loop knowledge update
# =========================================================

def update_knowledge_after_resolution(
    incident,
    root_cause,
    remediation,
    execution_result,
    validation_result,
    new_learning=False,
):

    # -----------------------------------------------------
    # Trusted-memory gate
    #
    # Do NOT save failed/unvalidated remediation.
    # -----------------------------------------------------

    if not validation_result.get(
        "validation_passed",
        False,
    ):

        return {
            "updated":
                False,

            "reason":
                (
                    "Incident validation did not pass. "
                    "Knowledge was not updated."
                ),
        }

    # -----------------------------------------------------
    # Episodic memory:
    # persist resolved historical incident
    # -----------------------------------------------------

    historical_record = (
        save_resolved_incident(
            incident=incident,
            root_cause=root_cause,
            remediation=remediation,
            execution_result=execution_result,
            validation_result=validation_result,
        )
    )

    # -----------------------------------------------------
    # Procedural memory:
    # update only when genuinely new learning exists
    # -----------------------------------------------------

    runbook_result = (
        update_runbook_if_needed(
            incident=incident,
            remediation=remediation,
            validation_result=validation_result,
            new_learning=new_learning,
        )
    )

    # -----------------------------------------------------
    # Refresh vector index
    # -----------------------------------------------------

    index_result = (
        refresh_knowledge_index()
    )

    return {
        "updated":
            True,

        "historical_incident":
            historical_record,

        "runbook":
            runbook_result,

        "index":
            index_result,
    }


# =========================================================
# Refresh vector index
# =========================================================

def refresh_knowledge_index():

    print(
        "\nRefreshing pgvector knowledge index..."
    )

    result = index_knowledge()

    return {
        "status":
            "INDEX_REFRESHED",

        "details":
            result,
    }


# =========================================================
# Standalone indexing entry point
# =========================================================

if __name__ == "__main__":

    result = index_knowledge()

    print(
        "\n========== INDEX RESULT =========="
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
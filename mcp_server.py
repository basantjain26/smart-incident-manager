from mcp.server.mcpserver import MCPServer

from diagnostic_tools import (
    get_job_logs as load_job_logs,
    get_job_metrics as load_job_metrics,
    get_schema_diff as load_schema_diff,
    detect_pipeline_anomaly as run_anomaly_detection,
    rerun_pipeline as run_pipeline,
    quarantine_partition as quarantine_data,
    rollback_deployment as rollback_code,
)

from incident_rag import search_knowledge as search_incident_knowledge


mcp = MCPServer("Smart Incident Manager")


@mcp.tool()
def get_job_logs(run_id: str) -> dict | None:
    """
    Get production error logs for a specific pipeline run.
    """

    return load_job_logs(run_id)


@mcp.tool()
def get_job_metrics(run_id: str) -> dict | None:
    """
    Get runtime and processing metrics for a pipeline run.
    """

    return load_job_metrics(run_id)


@mcp.tool()
def get_schema_diff(pipeline: str) -> dict | None:
    """
    Compare the previous and current schema for a pipeline.
    """

    return load_schema_diff(pipeline)


@mcp.tool()
def search_knowledge(
    query: str,
    top_k: int = 3,
    platform: str | None = None,
    environment: str | None = None,
    incident_type: str | None = None,
) -> list[dict]:
    """
    Search historical incidents and operational runbooks
    using semantic vector search.
    """

    return search_incident_knowledge(
        query=query,
        top_k=top_k,
        platform=platform,
        environment=environment,
        incident_type=incident_type,
    )

@mcp.tool()
def detect_pipeline_anomaly(
    pipeline: str,
    run_id: str,
) -> dict:
    """
    Detect unusual pipeline behavior using
    monitoring rules and Isolation Forest.
    """

    return run_anomaly_detection(
        pipeline,
        run_id,
    )

@mcp.tool()
def rerun_pipeline(
    pipeline: str,
    run_id: str,
) -> dict:
    """
    Propose rerunning a failed pipeline.
    This tool currently operates in dry-run mode only.
    """

    return run_pipeline(
        pipeline=pipeline,
        run_id=run_id,
        dry_run=True,
    )


@mcp.tool()
def quarantine_partition(
    pipeline: str,
    partition: str = "affected_batch",
) -> dict:
    """
    Propose quarantining an affected data partition.
    This tool currently operates in dry-run mode only.
    """

    return quarantine_data(
        pipeline=pipeline,
        partition=partition,
        dry_run=True,
    )


@mcp.tool()
def rollback_deployment(
    pipeline: str,
    version: str,
) -> dict:
    """
    Propose rolling back a pipeline deployment.
    This tool currently operates in dry-run mode only.
    """

    return rollback_code(
        pipeline=pipeline,
        version=version,
        dry_run=True,
    )

if __name__ == "__main__":
    mcp.run()
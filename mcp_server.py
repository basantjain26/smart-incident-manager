from mcp.server.mcpserver import MCPServer

from diagnostic_tools import (
    get_job_logs as load_job_logs,
    get_job_metrics as load_job_metrics,
    get_schema_diff as load_schema_diff,
    detect_pipeline_anomaly as run_anomaly_detection,
    rerun_pipeline as run_pipeline,
    quarantine_partition as quarantine_data,
    rollback_deployment as rollback_code,
    validate_remediation as run_validation,
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
    dry_run: bool = True,
) -> dict:
    """
    Rerun a failed pipeline.

    dry_run=True only proposes the action.
    dry_run=False executes the action.
    """

    return run_pipeline(
        pipeline=pipeline,
        run_id=run_id,
        dry_run=dry_run,
    )


@mcp.tool()
def quarantine_partition(
    pipeline: str,
    partition: str = "affected_batch",
    dry_run: bool = True,
) -> dict:
    """
    Quarantine an affected data partition or batch.

    dry_run=True only proposes the action.
    dry_run=False executes the action.
    """

    return quarantine_data(
        pipeline=pipeline,
        partition=partition,
        dry_run=dry_run,
    )


@mcp.tool()
def rollback_deployment(
    pipeline: str,
    version: str,
    dry_run: bool = True,
) -> dict:
    """
    Roll back a deployment.

    dry_run=True only proposes the action.
    dry_run=False executes the action.
    """

    return rollback_code(
        pipeline=pipeline,
        version=version,
        dry_run=dry_run,
    )

@mcp.tool()
def validate_remediation(
    recovery_run_id: str,
    expected_records: int = 10000000,
    duplicate_tolerance: int = 0,
) -> dict:
    """
    Validate pipeline and data health after remediation.
    """

    return run_validation(
        recovery_run_id=recovery_run_id,
        expected_records=expected_records,
        duplicate_tolerance=duplicate_tolerance,
    )

if __name__ == "__main__":
    mcp.run()
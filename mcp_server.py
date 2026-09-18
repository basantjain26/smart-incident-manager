from mcp.server.mcpserver import MCPServer

from diagnostic_tools import (
    get_job_logs as load_job_logs,
    get_job_metrics as load_job_metrics,
    get_schema_diff as load_schema_diff,
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

if __name__ == "__main__":
    mcp.run()
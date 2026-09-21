import asyncio
import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from mcp import (
    ClientSession,
    StdioServerParameters,
)
from mcp.client.stdio import (
    stdio_client,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from incident_manager import (
    load_incidents,
    convert_mcp_tools_to_openai,
    parse_mcp_result,
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

client = OpenAI()

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6",
)


# =========================================================
# Specialist tool permissions
#
# Important:
# Specialists receive only the tools relevant to
# their responsibility.
#
# No remediation tools are exposed here.
# =========================================================

SPECIALIST_TOOLS = {
    "SCHEMA": {
        "get_job_logs",
        "get_schema_diff",
        "search_knowledge",
    },

    "PERFORMANCE": {
        "get_job_metrics",
        "detect_pipeline_anomaly",
        "search_knowledge",
    },

    "DATA_QUALITY": {
        "get_job_logs",
        "get_job_metrics",
        "search_knowledge",
    },
}


# =========================================================
# Multi-Agent State
# =========================================================

class MultiAgentState(
    TypedDict,
    total=False,
):
    incident: dict

    # Supervisor decision
    route: str

    # Specialist output
    specialist_name: str
    specialist_analysis: str

    # Tools specialist actually used
    tool_trace: list

    # Final status
    final_status: str


# =========================================================
# Utility:
# Filter MCP tools for one specialist
# =========================================================

def filter_tools(
    all_mcp_tools,
    allowed_tool_names,
):

    filtered = [
        tool
        for tool in all_mcp_tools
        if tool.name
        in allowed_tool_names
    ]

    return (
        convert_mcp_tools_to_openai(
            filtered
        )
    )


# =========================================================
# Supervisor Agent
# =========================================================

def supervisor_node(
    state: MultiAgentState,
):

    print(
        "\n========== SUPERVISOR AGENT =========="
    )

    incident = state[
        "incident"
    ]

    instructions = """
You are the supervisor for a production data
engineering incident response system.

Your only responsibility is to route the incident
to the most appropriate specialist.

Available specialists:

SCHEMA
Use for:
- missing columns
- renamed columns
- incompatible data types
- schema evolution
- schema drift
- unresolved column errors

PERFORMANCE
Use for:
- unusually long runtime
- resource bottlenecks
- performance degradation
- abnormal freshness delay
- unusual runtime metrics
- anomaly detection

DATA_QUALITY
Use for:
- duplicate records
- row-count mismatch
- partial writes
- retry-created duplicates
- missing data
- unexpected business-data changes

Return ONLY one of these values:

SCHEMA
PERFORMANCE
DATA_QUALITY

Do not provide explanation.
"""

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": (
                    "Route this incident:\n\n"
                    + json.dumps(
                        incident,
                        indent=2,
                    )
                ),
            }
        ],
    )

    route = (
        response.output_text
        .strip()
        .upper()
        .replace(".", "")
    )

    # -----------------------------------------------------
    # Safety fallback
    #
    # Never allow arbitrary model text to become
    # a LangGraph route.
    # -----------------------------------------------------

    valid_routes = {
        "SCHEMA",
        "PERFORMANCE",
        "DATA_QUALITY",
    }

    if route not in valid_routes:

        route = (
            "DATA_QUALITY"
        )

    print(
        f"Supervisor selected: {route}"
    )

    return {
        "route": route
    }


# =========================================================
# LangGraph Router
# =========================================================

def specialist_router(
    state: MultiAgentState,
):

    route = state.get(
        "route"
    )

    if route == "SCHEMA":
        return "schema_agent"

    if route == "PERFORMANCE":
        return "performance_agent"

    return "data_quality_agent"


# =========================================================
# Generic Specialist Agent Runner
# =========================================================

async def run_specialist_agent(
    specialist_name,
    specialist_role,
    incident,
    mcp_session,
    tools,
):

    instructions = f"""
You are the {specialist_name} specialist in a
production data engineering incident-response system.

Your specialty is:

{specialist_role}

Your job is ONLY to investigate and diagnose.

You may use the provided MCP tools to gather evidence.

Rules:

1. Do not invent production facts.

2. Use tools when evidence is needed.

3. Current production evidence is stronger than
   historical incidents.

4. Historical incidents and runbooks are supporting
   evidence only.

5. Do not recommend or execute destructive actions.

6. You do NOT have permission to execute remediation.

7. Avoid unnecessary tool calls.

When finished, return:

1. Specialist Assessment
2. Evidence
3. Most Likely Root Cause
4. Confidence:
   LOW, MEDIUM, or HIGH
5. Recommended Next Investigation or Remediation Direction

Keep the answer concise and evidence-based.
"""

    input_items = [
        {
            "role": "user",
            "content": (
                "Investigate this incident:\n\n"
                + json.dumps(
                    incident,
                    indent=2,
                )
            ),
        }
    ]

    tool_trace = []

    while True:

        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=tools,
            input=input_items,
        )

        input_items.extend(
            response.output
        )

        tool_called = False

        for item in response.output:

            if (
                item.type
                != "function_call"
            ):
                continue

            tool_called = True

            arguments = json.loads(
                item.arguments
            )

            print(
                "\n--------------------------------"
            )

            print(
                f"{specialist_name} selected tool: "
                f"{item.name}"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2,
                )
            )

            tool_trace.append(
                {
                    "tool":
                        item.name,

                    "arguments":
                        arguments,
                }
            )

            # -------------------------------------------------
            # MCP executes the requested read-only tool.
            # -------------------------------------------------

            mcp_result = await (
                mcp_session.call_tool(
                    item.name,
                    arguments=arguments,
                )
            )

            result = (
                parse_mcp_result(
                    mcp_result
                )
            )

            print(
                "\nTool result:"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    default=str,
                )
            )

            input_items.append(
                {
                    "type":
                        "function_call_output",

                    "call_id":
                        item.call_id,

                    "output":
                        json.dumps(
                            result,
                            default=str,
                        ),
                }
            )

        # -----------------------------------------------------
        # No more tool calls means specialist investigation
        # is complete.
        # -----------------------------------------------------

        if not tool_called:

            return {
                "analysis":
                    response.output_text,

                "tool_trace":
                    tool_trace,
            }


# =========================================================
# Schema Specialist
# =========================================================

def create_schema_agent(
    mcp_session,
    all_mcp_tools,
):

    tools = filter_tools(
        all_mcp_tools,
        SPECIALIST_TOOLS[
            "SCHEMA"
        ],
    )

    async def schema_agent(
        state: MultiAgentState,
    ):

        print(
            "\n========== SCHEMA AGENT =========="
        )

        result = (
            await run_specialist_agent(
                specialist_name=
                    "Schema Agent",

                specialist_role=
                    """
Diagnose schema drift, renamed columns,
missing fields, incompatible data types,
schema evolution, and schema-related
pipeline failures.
""",

                incident=
                    state[
                        "incident"
                    ],

                mcp_session=
                    mcp_session,

                tools=
                    tools,
            )
        )

        return {
            "specialist_name":
                "SCHEMA",

            "specialist_analysis":
                result[
                    "analysis"
                ],

            "tool_trace":
                result[
                    "tool_trace"
                ],

            "final_status":
                "SPECIALIST_ANALYSIS_COMPLETE",
        }

    return schema_agent


# =========================================================
# Performance Specialist
# =========================================================

def create_performance_agent(
    mcp_session,
    all_mcp_tools,
):

    tools = filter_tools(
        all_mcp_tools,
        SPECIALIST_TOOLS[
            "PERFORMANCE"
        ],
    )

    async def performance_agent(
        state: MultiAgentState,
    ):

        print(
            "\n========== PERFORMANCE AGENT =========="
        )

        result = (
            await run_specialist_agent(
                specialist_name=
                    "Performance Agent",

                specialist_role=
                    """
Diagnose abnormal runtime, throughput
degradation, freshness delays, resource
problems, and anomalous operational metrics.
Use anomaly detection when appropriate.
""",

                incident=
                    state[
                        "incident"
                    ],

                mcp_session=
                    mcp_session,

                tools=
                    tools,
            )
        )

        return {
            "specialist_name":
                "PERFORMANCE",

            "specialist_analysis":
                result[
                    "analysis"
                ],

            "tool_trace":
                result[
                    "tool_trace"
                ],

            "final_status":
                "SPECIALIST_ANALYSIS_COMPLETE",
        }

    return performance_agent


# =========================================================
# Data Quality Specialist
# =========================================================

def create_data_quality_agent(
    mcp_session,
    all_mcp_tools,
):

    tools = filter_tools(
        all_mcp_tools,
        SPECIALIST_TOOLS[
            "DATA_QUALITY"
        ],
    )

    async def data_quality_agent(
        state: MultiAgentState,
    ):

        print(
            "\n========== DATA QUALITY AGENT =========="
        )

        result = (
            await run_specialist_agent(
                specialist_name=
                    "Data Quality Agent",

                specialist_role=
                    """
Diagnose duplicate records, partial writes,
row-count mismatches, missing data,
retry-related duplication, and unexpected
changes in business-data quality.
""",

                incident=
                    state[
                        "incident"
                    ],

                mcp_session=
                    mcp_session,

                tools=
                    tools,
            )
        )

        return {
            "specialist_name":
                "DATA_QUALITY",

            "specialist_analysis":
                result[
                    "analysis"
                ],

            "tool_trace":
                result[
                    "tool_trace"
                ],

            "final_status":
                "SPECIALIST_ANALYSIS_COMPLETE",
        }

    return data_quality_agent


# =========================================================
# Build Multi-Agent LangGraph
# =========================================================

def build_multi_agent_graph(
    mcp_session,
    all_mcp_tools,
):

    builder = StateGraph(
        MultiAgentState
    )

    # -----------------------------------------------------
    # Supervisor
    # -----------------------------------------------------

    builder.add_node(
        "supervisor",
        supervisor_node,
    )

    # -----------------------------------------------------
    # Specialist agents
    # -----------------------------------------------------

    builder.add_node(
        "schema_agent",
        create_schema_agent(
            mcp_session,
            all_mcp_tools,
        ),
    )

    builder.add_node(
        "performance_agent",
        create_performance_agent(
            mcp_session,
            all_mcp_tools,
        ),
    )

    builder.add_node(
        "data_quality_agent",
        create_data_quality_agent(
            mcp_session,
            all_mcp_tools,
        ),
    )

    # -----------------------------------------------------
    # Start with supervisor
    # -----------------------------------------------------

    builder.add_edge(
        START,
        "supervisor",
    )

    # -----------------------------------------------------
    # Supervisor dynamically selects specialist
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "supervisor",
        specialist_router,
        {
            "schema_agent":
                "schema_agent",

            "performance_agent":
                "performance_agent",

            "data_quality_agent":
                "data_quality_agent",
        },
    )

    # -----------------------------------------------------
    # Specialist completes investigation.
    # For now the graph ends here.
    #
    # Later this output can feed into our existing
    # full incident LangGraph workflow.
    # -----------------------------------------------------

    builder.add_edge(
        "schema_agent",
        END,
    )

    builder.add_edge(
        "performance_agent",
        END,
    )

    builder.add_edge(
        "data_quality_agent",
        END,
    )

    return builder.compile()


# =========================================================
# Run Multi-Agent Example
# =========================================================

async def run_multi_agent():

    incidents = (
        load_incidents()
    )

    # -----------------------------------------------------
    # Change this index to test different specialists:
    #
    # 0 -> Schema incident
    # 1 -> Duplicate/data-quality incident
    # 2 -> Performance anomaly
    # -----------------------------------------------------

    incident = incidents[
        1
    ]

    print(
        "\n========== INCIDENT =========="
    )

    print(
        json.dumps(
            incident,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Launch MCP server
    # -----------------------------------------------------

    server_params = (
        StdioServerParameters(
            command=
                sys.executable,

            args=[
                "mcp_server.py",
            ],

            env=dict(
                os.environ
            ),
        )
    )

    async with stdio_client(
        server_params
    ) as (
        read,
        write,
    ):

        async with ClientSession(
            read,
            write,
        ) as session:

            await session.initialize()

            # -------------------------------------------------
            # MCP discovers the full tool catalog.
            # Each specialist gets only its allowed subset.
            # -------------------------------------------------

            tools_response = (
                await session.list_tools()
            )

            print(
                "\n========== AVAILABLE MCP TOOLS =========="
            )

            for tool in (
                tools_response.tools
            ):

                print(
                    f"- {tool.name}"
                )

            graph = (
                build_multi_agent_graph(
                    mcp_session=
                        session,

                    all_mcp_tools=
                        tools_response.tools,
                )
            )

            initial_state = {
                "incident":
                    incident,

                "final_status":
                    "IN_PROGRESS",
            }

            result = (
                await graph.ainvoke(
                    initial_state
                )
            )

            # -------------------------------------------------
            # Final result
            # -------------------------------------------------

            print(
                "\n========================================"
            )

            print(
                "MULTI-AGENT INVESTIGATION COMPLETE"
            )

            print(
                "========================================"
            )

            print(
                "Supervisor Route:",
                result.get(
                    "route"
                ),
            )

            print(
                "Specialist:",
                result.get(
                    "specialist_name"
                ),
            )

            print(
                "\nSpecialist Analysis:"
            )

            print(
                result.get(
                    "specialist_analysis"
                )
            )

            print(
                "\nTool Trace:"
            )

            print(
                json.dumps(
                    result.get(
                        "tool_trace",
                        [],
                    ),
                    indent=2,
                )
            )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        run_multi_agent()
    )
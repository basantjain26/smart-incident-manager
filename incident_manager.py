import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import TypedDict

import psycopg
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
from langgraph.checkpoint.postgres import (
    PostgresSaver,
)
from langgraph.types import (
    interrupt,
    Command,
)

from knowledge_index import (
    update_knowledge_after_resolution,
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
# PostgreSQL configuration
# =========================================================

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


DB_URI = (
    f"postgresql://"
    f"{DB_USER}:"
    f"{DB_PASSWORD}"
    f"@{DB_HOST}:"
    f"{DB_PORT}/"
    f"{DB_NAME}"
)


# =========================================================
# Constants
# =========================================================

REMEDIATION_TOOLS = {
    "rerun_pipeline",
    "quarantine_partition",
    "rollback_deployment",
}


# =========================================================
# Audit table
# =========================================================

def create_audit_table():

    with psycopg.connect(
        DB_URI
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_audit (
                    id BIGSERIAL PRIMARY KEY,

                    incident_id TEXT NOT NULL,

                    event_type TEXT NOT NULL,

                    details JSONB,

                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )

        conn.commit()


# =========================================================
# Write audit event
# =========================================================

def write_audit_event(
    incident_id,
    event_type,
    details,
):

    with psycopg.connect(
        DB_URI
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO incident_audit (
                    incident_id,
                    event_type,
                    details,
                    created_at
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    incident_id,
                    event_type,
                    json.dumps(
                        details,
                        default=str,
                    ),
                    datetime.now(
                        timezone.utc
                    ),
                ),
            )

        conn.commit()


# =========================================================
# Load incidents
# =========================================================

def load_incidents():

    with open(
        "data/incidents.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =========================================================
# Convert MCP tools to OpenAI function tools
# =========================================================

def convert_mcp_tools_to_openai(
    mcp_tools,
):

    tools = []

    for tool in mcp_tools:

        tools.append(
            {
                "type":
                    "function",

                "name":
                    tool.name,

                "description":
                    tool.description
                    or "",

                "parameters":
                    tool.input_schema,
            }
        )

    return tools


# =========================================================
# Parse MCP result
# =========================================================

def parse_mcp_result(
    result,
):

    if getattr(
        result,
        "structuredContent",
        None,
    ) is not None:

        return result.structuredContent

    if getattr(
        result,
        "structured_content",
        None,
    ) is not None:

        return result.structured_content

    texts = []

    for item in result.content:

        if getattr(
            item,
            "text",
            None,
        ):

            texts.append(
                item.text
            )

    if not texts:
        return None

    combined = "\n".join(
        texts
    )

    try:

        return json.loads(
            combined
        )

    except json.JSONDecodeError:

        return combined


# =========================================================
# LangGraph state
# =========================================================

class RemediationState(
    TypedDict
):

    incident_id: str

    root_cause: str

    proposed_action: str

    action_arguments: dict

    approved: bool | None

    execution_result: dict | None


# =========================================================
# Human approval node
# =========================================================

def human_approval_node(
    state: RemediationState,
):

    decision = interrupt(
        {
            "incident_id":
                state[
                    "incident_id"
                ],

            "root_cause":
                state[
                    "root_cause"
                ],

            "proposed_action":
                state[
                    "proposed_action"
                ],

            "action_arguments":
                state[
                    "action_arguments"
                ],

            "message":
                (
                    "Review the proposed "
                    "remediation and approve "
                    "or reject execution."
                ),
        }
    )

    approved = (
        str(decision)
        .strip()
        .lower()
        in {
            "yes",
            "y",
            "approve",
            "approved",
        }
    )

    return {
        "approved":
            approved
    }


# =========================================================
# Approval router
# =========================================================

def approval_router(
    state: RemediationState,
):

    if state[
        "approved"
    ]:

        return "approved"

    return "rejected"


# =========================================================
# Approved node
# =========================================================

def approved_node(
    state: RemediationState,
):

    return {
        "execution_result": {
            "status":
                "APPROVED",

            "action":
                state[
                    "proposed_action"
                ],

            "arguments":
                state[
                    "action_arguments"
                ],

            "message":
                (
                    "Human approved remediation. "
                    "Ready for MCP execution."
                ),
        }
    }


# =========================================================
# Rejected node
# =========================================================

def rejected_node(
    state: RemediationState,
):

    return {
        "execution_result": {
            "status":
                "REJECTED",

            "action":
                state[
                    "proposed_action"
                ],

            "arguments":
                state[
                    "action_arguments"
                ],

            "message":
                (
                    "Human reviewer rejected "
                    "the proposed remediation."
                ),
        }
    }


# =========================================================
# Build graph definition
#
# Checkpointer is passed in so its DB connection
# remains open for the graph lifetime.
# =========================================================

def build_remediation_graph(
    checkpointer,
):

    builder = StateGraph(
        RemediationState
    )

    builder.add_node(
        "human_approval",
        human_approval_node,
    )

    builder.add_node(
        "approved",
        approved_node,
    )

    builder.add_node(
        "rejected",
        rejected_node,
    )

    builder.add_edge(
        START,
        "human_approval",
    )

    builder.add_conditional_edges(
        "human_approval",
        approval_router,
        {
            "approved":
                "approved",

            "rejected":
                "rejected",
        },
    )

    builder.add_edge(
        "approved",
        END,
    )

    builder.add_edge(
        "rejected",
        END,
    )

    return builder.compile(
        checkpointer=
            checkpointer
    )


# =========================================================
# Human-in-the-loop approval
#
# PostgreSQL checkpointer survives process restarts.
# Same thread_id identifies the workflow.
# =========================================================

def run_human_approval(
    incident_id,
    root_cause,
    proposed_action,
    action_arguments,
):

    config = {
        "configurable": {
            "thread_id":
                incident_id
        }
    }

    state = {
        "incident_id":
            incident_id,

        "root_cause":
            root_cause,

        "proposed_action":
            proposed_action,

        "action_arguments":
            action_arguments,

        "approved":
            None,

        "execution_result":
            None,
    }

    # -----------------------------------------------------
    # Keep saver connection open during graph invocation.
    # -----------------------------------------------------

    with PostgresSaver.from_conn_string(
        DB_URI
    ) as checkpointer:

        # For this small project we call setup here.
        # In a real deployment this would normally be
        # performed once during deployment/migrations.

        checkpointer.setup()

        graph = (
            build_remediation_graph(
                checkpointer
            )
        )

        # -------------------------------------------------
        # Execute until interrupt()
        # -------------------------------------------------

        result = graph.invoke(
            state,
            config=config,
        )

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if not interrupts:

            return result

        print(
            "\n========== HUMAN APPROVAL REQUIRED =========="
        )

        for pending in interrupts:

            approval_data = getattr(
                pending,
                "value",
                pending,
            )

            print(
                json.dumps(
                    approval_data,
                    indent=2,
                    default=str,
                )
            )

        decision = input(
            "\nApprove remediation? "
            "(yes/no): "
        )

        # -------------------------------------------------
        # Resume same persistent thread
        # -------------------------------------------------

        result = graph.invoke(
            Command(
                resume=decision
            ),
            config=config,
        )

        return result


# =========================================================
# Execute approved remediation
# =========================================================

async def execute_approved_remediation(
    mcp_session,
    proposed_action,
    action_arguments,
):

    execution_arguments = dict(
        action_arguments
    )

    # -----------------------------------------------------
    # Human approval is the boundary where
    # dry_run changes from True → False.
    # -----------------------------------------------------

    execution_arguments[
        "dry_run"
    ] = False

    print(
        "\n========== EXECUTING APPROVED ACTION =========="
    )

    print(
        f"Action: "
        f"{proposed_action}"
    )

    print(
        "\nArguments:"
    )

    print(
        json.dumps(
            execution_arguments,
            indent=2,
        )
    )

    mcp_result = await (
        mcp_session.call_tool(
            proposed_action,
            arguments=
                execution_arguments,
        )
    )

    result = (
        parse_mcp_result(
            mcp_result
        )
    )

    print(
        "\nMCP execution result:"
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    return result


# =========================================================
# Post-remediation validation
# =========================================================

async def validate_after_remediation(
    mcp_session,
    recovery_run_id,
    expected_records,
    duplicate_tolerance=0,
):

    print(
        "\n========== POST-REMEDIATION VALIDATION =========="
    )

    mcp_result = await (
        mcp_session.call_tool(
            "validate_remediation",
            arguments={
                "recovery_run_id":
                    recovery_run_id,

                "expected_records":
                    expected_records,

                "duplicate_tolerance":
                    duplicate_tolerance,
            },
        )
    )

    validation_result = (
        parse_mcp_result(
            mcp_result
        )
    )

    print(
        json.dumps(
            validation_result,
            indent=2,
            default=str,
        )
    )

    return validation_result


# =========================================================
# Main incident investigation agent
# =========================================================

async def investigate_incident(
    incident,
    mcp_session,
    llm_tools,
):

    instructions = """
You are a production data engineering incident investigator.

Your objectives are to:

1. Investigate the production incident.
2. Identify the most likely root cause.
3. Gather current operational evidence.
4. Search historical incidents and approved runbooks when useful.
5. Determine downstream impact.
6. Recommend the safest remediation.
7. Call an appropriate remediation tool in DRY-RUN mode
   when sufficient evidence supports the action.

You have access to diagnostic, anomaly-detection,
knowledge-search, validation, and remediation tools
through MCP.

IMPORTANT RULES:

1. Do not invent production facts.

2. Use diagnostic tools to collect current operational evidence.

3. Historical incidents and runbooks are supporting evidence.
   They do not prove the current root cause.

4. Prefer current production evidence when it conflicts
   with historical knowledge.

5. Do not call tools unnecessarily.

6. Use search_knowledge when previous incidents or
   approved runbooks can help.

7. Remediation tools MUST ONLY be called with dry_run=true
   during investigation.

8. Never execute production remediation directly.

9. When the root cause is sufficiently supported and an
   appropriate remediation tool exists, call that remediation
   tool in dry-run mode.

10. Prefer the least risky remediation capable of safely
    restoring the system.

11. A dry-run remediation is only a proposal.
    It is not execution.

12. Human approval and application logic control
    real remediation execution.

When investigation is complete provide:

1. Root Cause
2. Current Evidence
3. Relevant Historical Knowledge
4. Impact
5. Recommended Remediation
6. Validation Steps
"""

    input_items = [
        {
            "role":
                "user",

            "content":
                (
                    "Investigate this production "
                    "incident:\n\n"
                    + json.dumps(
                        incident,
                        indent=2,
                    )
                ),
        }
    ]

    proposed_remediation = None

    tool_trace = []

    retrieved_documents = []

    safety_violations = []

    # =====================================================
    # Telemetry
    # =====================================================

    investigation_start = (
        time.perf_counter()
    )

    llm_latency_seconds = 0.0

    tool_latency_seconds = 0.0

    llm_calls = 0

    mcp_tool_calls = 0

    input_tokens = 0

    output_tokens = 0

    total_tokens = 0


    while True:

        # -------------------------------------------------
        # LLM call
        # -------------------------------------------------

        llm_start = (
            time.perf_counter()
        )

        response = (
            client.responses.create(
                model=
                    MODEL,

                instructions=
                    instructions,

                tools=
                    llm_tools,

                input=
                    input_items,
            )
        )

        llm_elapsed = (
            time.perf_counter()
            - llm_start
        )

        llm_latency_seconds += (
            llm_elapsed
        )

        llm_calls += 1


        # -------------------------------------------------
        # Token usage
        # -------------------------------------------------

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage:

            current_input_tokens = (
                getattr(
                    usage,
                    "input_tokens",
                    0,
                )
                or 0
            )

            current_output_tokens = (
                getattr(
                    usage,
                    "output_tokens",
                    0,
                )
                or 0
            )

            current_total_tokens = (
                getattr(
                    usage,
                    "total_tokens",
                    0,
                )
                or (
                    current_input_tokens
                    + current_output_tokens
                )
            )

            input_tokens += (
                current_input_tokens
            )

            output_tokens += (
                current_output_tokens
            )

            total_tokens += (
                current_total_tokens
            )


        input_items.extend(
            response.output
        )

        tool_called = False


        # -------------------------------------------------
        # Process tool calls
        # -------------------------------------------------

        for item in response.output:

            if (
                item.type
                != "function_call"
            ):

                continue

            tool_called = True

            arguments = (
                json.loads(
                    item.arguments
                )
            )

            original_arguments = dict(
                arguments
            )

            tool_trace.append(
                {
                    "tool":
                        item.name,

                    "arguments":
                        original_arguments,
                }
            )

            print(
                "\n--------------------------------"
            )

            print(
                "Agent selected MCP tool: "
                f"{item.name}"
            )

            print(
                "\nArguments before guardrails:"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2,
                )
            )


            # =================================================
            # Hard remediation safety guardrail
            # =================================================

            if (
                item.name
                in REMEDIATION_TOOLS
            ):

                if (
                    original_arguments.get(
                        "dry_run"
                    )
                    is False
                ):

                    safety_violations.append(
                        {
                            "tool":
                                item.name,

                            "arguments":
                                original_arguments,

                            "reason":
                                (
                                    "Model attempted remediation "
                                    "with dry_run=False during "
                                    "investigation."
                                ),
                        }
                    )

                arguments[
                    "dry_run"
                ] = True

                print(
                    "\nGuardrail applied: "
                    "dry_run forced to True"
                )


            print(
                "\nFinal tool arguments:"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2,
                )
            )


            # -------------------------------------------------
            # MCP execution + latency
            # -------------------------------------------------

            tool_start = (
                time.perf_counter()
            )

            mcp_result = await (
                mcp_session.call_tool(
                    item.name,
                    arguments=
                        arguments,
                )
            )

            tool_elapsed = (
                time.perf_counter()
                - tool_start
            )

            tool_latency_seconds += (
                tool_elapsed
            )

            mcp_tool_calls += 1


            result = (
                parse_mcp_result(
                    mcp_result
                )
            )


            print(
                "\nMCP tool result:"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    default=str,
                )
            )


            # -------------------------------------------------
            # Capture RAG documents
            # -------------------------------------------------

            if (
                item.name
                == "search_knowledge"
                and isinstance(
                    result,
                    list,
                )
            ):

                for row in result:

                    if not isinstance(
                        row,
                        dict,
                    ):

                        continue

                    document_id = (
                        row.get(
                            "document_id"
                        )
                    )

                    if (
                        document_id
                        and document_id
                        not in retrieved_documents
                    ):

                        retrieved_documents.append(
                            document_id
                        )


            # -------------------------------------------------
            # Capture dry-run remediation proposal
            # -------------------------------------------------

            if (
                item.name
                in REMEDIATION_TOOLS
            ):

                proposed_remediation = {
                    "action":
                        item.name,

                    "arguments":
                        arguments,

                    "dry_run_result":
                        result,
                }


            # -------------------------------------------------
            # Send tool result back to LLM
            # -------------------------------------------------

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


        # -------------------------------------------------
        # Investigation complete
        # -------------------------------------------------

        if not tool_called:

            investigation_latency_seconds = (
                time.perf_counter()
                - investigation_start
            )

            return {
                "final_rca":
                    response.output_text,

                "root_cause_summary":
                    response.output_text,

                "proposed_remediation":
                    proposed_remediation,

                "tool_trace":
                    tool_trace,

                "retrieved_documents":
                    retrieved_documents,

                "safety_violations":
                    safety_violations,

                "telemetry": {
                    "model":
                        MODEL,

                    "llm_calls":
                        llm_calls,

                    "mcp_tool_calls":
                        mcp_tool_calls,

                    "input_tokens":
                        input_tokens,

                    "output_tokens":
                        output_tokens,

                    "total_tokens":
                        total_tokens,

                    "investigation_latency_seconds":
                        round(
                            investigation_latency_seconds,
                            3,
                        ),

                    "llm_latency_seconds":
                        round(
                            llm_latency_seconds,
                            3,
                        ),

                    "tool_latency_seconds":
                        round(
                            tool_latency_seconds,
                            3,
                        ),
                },
            }


# =========================================================
# Main
# =========================================================

async def main():

    # -----------------------------------------------------
    # Initialize audit storage
    # -----------------------------------------------------

    create_audit_table()


    incidents = (
        load_incidents()
    )

    # Duplicate/retry demo incident
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

            # =============================================
            # MCP initialization
            # =============================================

            await session.initialize()


            # =============================================
            # Dynamic MCP tool discovery
            # =============================================

            tools_response = (
                await session.list_tools()
            )


            print(
                "\n========== MCP TOOLS =========="
            )

            for tool in (
                tools_response.tools
            ):

                print(
                    f"- {tool.name}"
                )


            llm_tools = (
                convert_mcp_tools_to_openai(
                    tools_response.tools
                )
            )


            # =============================================
            # Investigation
            # =============================================

            investigation = (
                await investigate_incident(
                    incident=
                        incident,

                    mcp_session=
                        session,

                    llm_tools=
                        llm_tools,
                )
            )


            print(
                "\n========== FINAL RCA =========="
            )

            print(
                investigation[
                    "final_rca"
                ]
            )


            print(
                "\n========== TELEMETRY =========="
            )

            print(
                json.dumps(
                    investigation[
                        "telemetry"
                    ],
                    indent=2,
                )
            )


            # =============================================
            # Remediation proposal
            # =============================================

            proposed = (
                investigation[
                    "proposed_remediation"
                ]
            )


            if proposed is None:

                print(
                    "\nNo automated remediation "
                    "was proposed by the agent."
                )

                return


            print(
                "\n========== PROPOSED REMEDIATION =========="
            )

            print(
                json.dumps(
                    proposed,
                    indent=2,
                    default=str,
                )
            )


            # =============================================
            # AUDIT: proposal
            # =============================================

            write_audit_event(
                incident_id=
                    incident[
                        "incident_id"
                    ],

                event_type=
                    "REMEDIATION_PROPOSED",

                details=
                    proposed,
            )


            # =============================================
            # Human approval through LangGraph
            # =============================================

            approval_result = (
                run_human_approval(
                    incident_id=
                        incident[
                            "incident_id"
                        ],

                    root_cause=
                        investigation[
                            "root_cause_summary"
                        ],

                    proposed_action=
                        proposed[
                            "action"
                        ],

                    action_arguments=
                        proposed[
                            "arguments"
                        ],
                )
            )


            approval_execution_result = (
                approval_result.get(
                    "execution_result",
                    {},
                )
            )


            approval_status = (
                approval_execution_result.get(
                    "status"
                )
            )


            print(
                "\n========== APPROVAL RESULT =========="
            )

            print(
                json.dumps(
                    approval_execution_result,
                    indent=2,
                    default=str,
                )
            )


            # =============================================
            # AUDIT: approval/rejection
            # =============================================

            if (
                approval_status
                == "APPROVED"
            ):

                audit_event_type = (
                    "REMEDIATION_APPROVED"
                )

            else:

                audit_event_type = (
                    "REMEDIATION_REJECTED"
                )


            write_audit_event(
                incident_id=
                    incident[
                        "incident_id"
                    ],

                event_type=
                    audit_event_type,

                details=
                    approval_execution_result,
            )


            # =============================================
            # Rejected
            # =============================================

            if (
                approval_status
                != "APPROVED"
            ):

                print(
                    "\nRemediation was not approved."
                )

                print(
                    "No production action "
                    "will be executed."
                )

                return


            # =============================================
            # Execute approved action
            # =============================================

            execution_result = (
                await execute_approved_remediation(
                    mcp_session=
                        session,

                    proposed_action=
                        proposed[
                            "action"
                        ],

                    action_arguments=
                        proposed[
                            "arguments"
                        ],
                )
            )


            print(
                "\n========== FINAL EXECUTION RESULT =========="
            )

            print(
                json.dumps(
                    execution_result,
                    indent=2,
                    default=str,
                )
            )


            # =============================================
            # AUDIT: execution
            # =============================================

            write_audit_event(
                incident_id=
                    incident[
                        "incident_id"
                    ],

                event_type=
                    "REMEDIATION_EXECUTED",

                details=
                    execution_result,
            )


            # =============================================
            # Check execution
            # =============================================

            if (
                not isinstance(
                    execution_result,
                    dict,
                )
                or execution_result.get(
                    "status"
                )
                != "EXECUTED"
            ):

                print(
                    "\nRemediation execution failed."
                )

                print(
                    "Incident remains OPEN."
                )

                return


            # =============================================
            # Recovery run
            # =============================================

            recovery_run_id = (
                "RUN-1002-RECOVERY"
            )


            # =============================================
            # Validation
            # =============================================

            validation_result = (
                await validate_after_remediation(
                    mcp_session=
                        session,

                    recovery_run_id=
                        recovery_run_id,

                    expected_records=
                        10_000_000,

                    duplicate_tolerance=
                        0,
                )
            )


            # =============================================
            # AUDIT: validation
            # =============================================

            write_audit_event(
                incident_id=
                    incident[
                        "incident_id"
                    ],

                event_type=
                    "REMEDIATION_VALIDATED",

                details=
                    validation_result,
            )


            # =============================================
            # Validation failed
            # =============================================

            if (
                not isinstance(
                    validation_result,
                    dict,
                )
                or not validation_result.get(
                    "validation_passed",
                    False,
                )
            ):

                print(
                    "\n========================================"
                )

                print(
                    "VALIDATION FAILED"
                )

                print(
                    "Incident remains OPEN."
                )

                print(
                    "========================================"
                )

                return


            # =============================================
            # Incident resolved
            # =============================================

            print(
                "\n========================================"
            )

            print(
                "INCIDENT RESOLVED"
            )

            print(
                "========================================"
            )


            # =============================================
            # Closed-loop knowledge update
            # =============================================

            print(
                "\n========== CLOSED-LOOP KNOWLEDGE UPDATE =========="
            )


            knowledge_update = (
                update_knowledge_after_resolution(
                    incident=
                        incident,

                    root_cause=
                        investigation[
                            "root_cause_summary"
                        ],

                    remediation=
                        proposed,

                    execution_result=
                        execution_result,

                    validation_result=
                        validation_result,

                    new_learning=
                        False,
                )
            )


            print(
                json.dumps(
                    knowledge_update,
                    indent=2,
                    default=str,
                )
            )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
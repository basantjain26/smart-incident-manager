import asyncio
import json
import os
import sys
from typing import TypedDict, Any

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

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from langgraph.types import (
    interrupt,
    Command,
)

from incident_manager import (
    load_incidents,
    convert_mcp_tools_to_openai,
    investigate_incident,
    execute_approved_remediation,
    validate_after_remediation,
)

from knowledge_index import (
    update_knowledge_after_resolution,
)


# =========================================================
# LangGraph State
# =========================================================

class IncidentWorkflowState(
    TypedDict,
    total=False,
):
    # Original production incident
    incident: dict

    # Result returned by the incident investigation agent
    investigation: dict

    # Structured remediation proposal
    proposed_remediation: dict | None

    # Human approval result
    approved: bool | None

    # Result from actual MCP remediation execution
    execution_result: dict | None

    # Configuration used for deterministic validation
    validation_config: dict

    # Result of post-remediation validation
    validation_result: dict | None

    # Result of closed-loop knowledge update
    knowledge_update: dict | None

    # Final workflow status
    final_status: str


# =========================================================
# Node 1
# Investigate incident
# =========================================================

def create_investigation_node(
    mcp_session,
    llm_tools,
):

    async def investigation_node(
        state: IncidentWorkflowState,
    ):

        print(
            "\n========== LANGGRAPH: INVESTIGATE =========="
        )

        incident = state[
            "incident"
        ]

        investigation = (
            await investigate_incident(
                incident=incident,
                mcp_session=mcp_session,
                llm_tools=llm_tools,
            )
        )

        proposed_remediation = (
            investigation.get(
                "proposed_remediation"
            )
        )

        return {
            "investigation":
                investigation,

            "proposed_remediation":
                proposed_remediation,
        }

    return investigation_node


# =========================================================
# Router 1
# Did the agent propose remediation?
# =========================================================

def remediation_router(
    state: IncidentWorkflowState,
):

    proposed = state.get(
        "proposed_remediation"
    )

    if proposed:
        return "approval"

    return "no_remediation"


# =========================================================
# Node 2
# No remediation available
# =========================================================

def no_remediation_node(
    state: IncidentWorkflowState,
):

    print(
        "\n========== LANGGRAPH: NO REMEDIATION =========="
    )

    return {
        "final_status":
            "MANUAL_REVIEW_REQUIRED"
    }


# =========================================================
# Node 3
# Human approval
# =========================================================

def human_approval_node(
    state: IncidentWorkflowState,
):

    proposed = state[
        "proposed_remediation"
    ]

    investigation = state[
        "investigation"
    ]

    incident = state[
        "incident"
    ]

    # -----------------------------------------------------
    # interrupt() pauses the graph here.
    #
    # Nothing is executed yet.
    # -----------------------------------------------------

    decision = interrupt(
        {
            "incident_id":
                incident.get(
                    "incident_id"
                ),

            "root_cause":
                investigation.get(
                    "root_cause_summary"
                ),

            "proposed_action":
                proposed.get(
                    "action"
                ),

            "action_arguments":
                proposed.get(
                    "arguments"
                ),

            "message":
                (
                    "Review the proposed remediation "
                    "and approve or reject execution."
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
# Router 2
# Human approved or rejected?
# =========================================================

def approval_router(
    state: IncidentWorkflowState,
):

    if state.get(
        "approved"
    ):
        return "execute"

    return "rejected"


# =========================================================
# Node 4
# Rejected remediation
# =========================================================

def rejected_node(
    state: IncidentWorkflowState,
):

    print(
        "\n========== LANGGRAPH: REMEDIATION REJECTED =========="
    )

    return {
        "final_status":
            "REMEDIATION_REJECTED"
    }


# =========================================================
# Node 5
# Execute approved remediation
# =========================================================

def create_execution_node(
    mcp_session,
):

    async def execution_node(
        state: IncidentWorkflowState,
    ):

        print(
            "\n========== LANGGRAPH: EXECUTE =========="
        )

        proposed = state[
            "proposed_remediation"
        ]

        execution_result = (
            await execute_approved_remediation(
                mcp_session=
                    mcp_session,

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

        return {
            "execution_result":
                execution_result
        }

    return execution_node


# =========================================================
# Router 3
# Did execution succeed?
# =========================================================

def execution_router(
    state: IncidentWorkflowState,
):

    result = state.get(
        "execution_result"
    )

    if (
        isinstance(
            result,
            dict,
        )
        and result.get(
            "status"
        )
        == "EXECUTED"
    ):
        return "validate"

    return "execution_failed"


# =========================================================
# Node 6
# Execution failure
# =========================================================

def execution_failed_node(
    state: IncidentWorkflowState,
):

    print(
        "\n========== LANGGRAPH: EXECUTION FAILED =========="
    )

    return {
        "final_status":
            "EXECUTION_FAILED"
    }


# =========================================================
# Node 7
# Post-remediation validation
# =========================================================

def create_validation_node(
    mcp_session,
):

    async def validation_node(
        state: IncidentWorkflowState,
    ):

        print(
            "\n========== LANGGRAPH: VALIDATE =========="
        )

        validation_config = state.get(
            "validation_config",
            {},
        )

        recovery_run_id = (
            validation_config.get(
                "recovery_run_id"
            )
        )

        expected_records = (
            validation_config.get(
                "expected_records"
            )
        )

        duplicate_tolerance = (
            validation_config.get(
                "duplicate_tolerance",
                0,
            )
        )

        # ---------------------------------------------
        # Fail safely when required validation
        # information is missing.
        # ---------------------------------------------

        if not recovery_run_id:

            return {
                "validation_result": {
                    "validation_passed":
                        False,

                    "reason":
                        (
                            "Missing recovery_run_id "
                            "for validation."
                        ),
                }
            }

        validation_result = (
            await validate_after_remediation(
                mcp_session=
                    mcp_session,

                recovery_run_id=
                    recovery_run_id,

                expected_records=
                    expected_records,

                duplicate_tolerance=
                    duplicate_tolerance,
            )
        )

        return {
            "validation_result":
                validation_result
        }

    return validation_node


# =========================================================
# Router 4
# Did validation succeed?
# =========================================================

def validation_router(
    state: IncidentWorkflowState,
):

    result = state.get(
        "validation_result"
    )

    if (
        isinstance(
            result,
            dict,
        )
        and result.get(
            "validation_passed",
            False,
        )
    ):
        return "update_knowledge"

    return "validation_failed"


# =========================================================
# Node 8
# Validation failed
# =========================================================

def validation_failed_node(
    state: IncidentWorkflowState,
):

    print(
        "\n========== LANGGRAPH: VALIDATION FAILED =========="
    )

    return {
        "final_status":
            "VALIDATION_FAILED"
    }


# =========================================================
# Node 9
# Closed-loop knowledge update
# =========================================================

def knowledge_update_node(
    state: IncidentWorkflowState,
):

    print(
        "\n========== LANGGRAPH: UPDATE KNOWLEDGE =========="
    )

    incident = state[
        "incident"
    ]

    investigation = state[
        "investigation"
    ]

    proposed = state[
        "proposed_remediation"
    ]

    execution_result = state[
        "execution_result"
    ]

    validation_result = state[
        "validation_result"
    ]

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

            # Existing runbook already covers
            # duplicate/retry pattern.
            new_learning=
                False,
        )
    )

    return {
        "knowledge_update":
            knowledge_update,

        "final_status":
            "RESOLVED",
    }


# =========================================================
# Build Full LangGraph
# =========================================================

def build_incident_graph(
    mcp_session,
    llm_tools,
):

    builder = StateGraph(
        IncidentWorkflowState
    )

    # -----------------------------------------------------
    # Register nodes
    # -----------------------------------------------------

    builder.add_node(
        "investigate",
        create_investigation_node(
            mcp_session,
            llm_tools,
        ),
    )

    builder.add_node(
        "human_approval",
        human_approval_node,
    )

    builder.add_node(
        "execute",
        create_execution_node(
            mcp_session
        ),
    )

    builder.add_node(
        "validate",
        create_validation_node(
            mcp_session
        ),
    )

    builder.add_node(
        "update_knowledge",
        knowledge_update_node,
    )

    builder.add_node(
        "no_remediation",
        no_remediation_node,
    )

    builder.add_node(
        "rejected",
        rejected_node,
    )

    builder.add_node(
        "execution_failed",
        execution_failed_node,
    )

    builder.add_node(
        "validation_failed",
        validation_failed_node,
    )

    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    builder.add_edge(
        START,
        "investigate",
    )

    # -----------------------------------------------------
    # Investigation routing
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "investigate",
        remediation_router,
        {
            "approval":
                "human_approval",

            "no_remediation":
                "no_remediation",
        },
    )

    # -----------------------------------------------------
    # Human approval routing
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "human_approval",
        approval_router,
        {
            "execute":
                "execute",

            "rejected":
                "rejected",
        },
    )

    # -----------------------------------------------------
    # Execution routing
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "execute",
        execution_router,
        {
            "validate":
                "validate",

            "execution_failed":
                "execution_failed",
        },
    )

    # -----------------------------------------------------
    # Validation routing
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "validate",
        validation_router,
        {
            "update_knowledge":
                "update_knowledge",

            "validation_failed":
                "validation_failed",
        },
    )

    # -----------------------------------------------------
    # Terminal paths
    # -----------------------------------------------------

    builder.add_edge(
        "update_knowledge",
        END,
    )

    builder.add_edge(
        "no_remediation",
        END,
    )

    builder.add_edge(
        "rejected",
        END,
    )

    builder.add_edge(
        "execution_failed",
        END,
    )

    builder.add_edge(
        "validation_failed",
        END,
    )

    # -----------------------------------------------------
    # Small/local project:
    # In-memory checkpoint is enough for now.
    #
    # We can swap this with PostgresSaver later without
    # redesigning the graph.
    # -----------------------------------------------------

    checkpointer = (
        InMemorySaver()
    )

    return builder.compile(
        checkpointer=
            checkpointer
    )


# =========================================================
# Run workflow
# =========================================================

async def run_workflow():

    incidents = (
        load_incidents()
    )

    # -----------------------------------------------------
    # Demo:
    # duplicate/retry incident
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
    # Start MCP server
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
            # Dynamic MCP tool discovery
            # -------------------------------------------------

            tools_response = (
                await session.list_tools()
            )

            llm_tools = (
                convert_mcp_tools_to_openai(
                    tools_response.tools
                )
            )

            # -------------------------------------------------
            # Build full workflow
            # -------------------------------------------------

            graph = (
                build_incident_graph(
                    mcp_session=
                        session,

                    llm_tools=
                        llm_tools,
                )
            )

            # -------------------------------------------------
            # Thread ID is important because LangGraph uses
            # it to associate checkpoints with this workflow.
            # -------------------------------------------------

            config = {
                "configurable": {
                    "thread_id":
                        incident[
                            "incident_id"
                        ]
                }
            }

            # -------------------------------------------------
            # Initial graph state
            # -------------------------------------------------

            initial_state = {
                "incident":
                    incident,

                "approved":
                    None,

                "final_status":
                    "IN_PROGRESS",

                # -----------------------------------------
                # For our simulated INC-1002 recovery.
                #
                # In production these values would usually
                # come from the remediation/orchestrator.
                # -----------------------------------------

                "validation_config": {
                    "recovery_run_id":
                        "RUN-1002-RECOVERY",

                    "expected_records":
                        10_000_000,

                    "duplicate_tolerance":
                        0,
                },
            }

            # -------------------------------------------------
            # Run until graph reaches END or interrupt()
            # -------------------------------------------------

            result = await graph.ainvoke(
                initial_state,
                config=config,
            )

            # -------------------------------------------------
            # Did graph pause for human approval?
            # -------------------------------------------------

            interrupts = result.get(
                "__interrupt__",
                [],
            )

            if interrupts:

                print(
                    "\n========================================"
                )

                print(
                    "LANGGRAPH PAUSED FOR HUMAN APPROVAL"
                )

                print(
                    "========================================"
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

                # -----------------------------------------
                # Resume SAME graph/thread.
                # -----------------------------------------

                result = await graph.ainvoke(
                    Command(
                        resume=
                            decision
                    ),
                    config=config,
                )

            # -------------------------------------------------
            # Final workflow result
            # -------------------------------------------------

            print(
                "\n========================================"
            )

            print(
                "LANGGRAPH WORKFLOW COMPLETE"
            )

            print(
                "========================================"
            )

            print(
                "Final Status:",
                result.get(
                    "final_status"
                ),
            )

            print(
                "\nExecution Result:"
            )

            print(
                json.dumps(
                    result.get(
                        "execution_result"
                    ),
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nValidation Result:"
            )

            print(
                json.dumps(
                    result.get(
                        "validation_result"
                    ),
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nKnowledge Update:"
            )

            print(
                json.dumps(
                    result.get(
                        "knowledge_update"
                    ),
                    indent=2,
                    default=str,
                )
            )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        run_workflow()
    )
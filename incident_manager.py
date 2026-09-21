import asyncio
import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


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
# Constants
# =========================================================

REMEDIATION_TOOLS = {
    "rerun_pipeline",
    "quarantine_partition",
    "rollback_deployment",
}


# =========================================================
# Load incidents
# =========================================================

def load_incidents():
    with open("data/incidents.json") as f:
        return json.load(f)


# =========================================================
# Convert MCP tools into OpenAI tool definitions
# =========================================================

def convert_mcp_tools_to_openai(mcp_tools):
    tools = []

    for tool in mcp_tools:
        tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
        )

    return tools


# =========================================================
# Convert MCP result into regular Python data
# =========================================================

def parse_mcp_result(result):

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
        if getattr(item, "text", None):
            texts.append(item.text)

    if not texts:
        return None

    combined = "\n".join(texts)

    try:
        return json.loads(combined)

    except json.JSONDecodeError:
        return combined


# =========================================================
# LangGraph remediation state
# =========================================================

class RemediationState(TypedDict):
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
                state["incident_id"],

            "root_cause":
                state["root_cause"],

            "proposed_action":
                state["proposed_action"],

            "action_arguments":
                state["action_arguments"],

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
        "approved": approved
    }


# =========================================================
# Route approval
# =========================================================

def approval_router(
    state: RemediationState,
):

    if state["approved"]:
        return "approved"

    return "rejected"


# =========================================================
# Approved node
#
# IMPORTANT:
# Approval does NOT execute remediation.
# It only authorizes later MCP execution.
# =========================================================

def approved_node(
    state: RemediationState,
):

    return {
        "execution_result": {
            "status": "APPROVED",

            "action":
                state["proposed_action"],

            "arguments":
                state["action_arguments"],

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
            "status": "REJECTED",

            "action":
                state["proposed_action"],

            "arguments":
                state["action_arguments"],

            "message":
                (
                    "Human reviewer rejected "
                    "the proposed remediation."
                ),
        }
    }


# =========================================================
# Build LangGraph approval workflow
# =========================================================

def build_remediation_graph():

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
            "approved": "approved",
            "rejected": "rejected",
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

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# =========================================================
# Run human approval
# =========================================================

def run_human_approval(
    incident_id,
    root_cause,
    proposed_action,
    action_arguments,
):

    graph = build_remediation_graph()

    config = {
        "configurable": {
            "thread_id": incident_id
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
    # Run graph until interrupt()
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Resume same LangGraph thread
    # -----------------------------------------------------

    result = graph.invoke(
        Command(
            resume=decision
        ),
        config=config,
    )

    return result


# =========================================================
# Execute approved remediation through MCP
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
    # Authorization boundary:
    #
    # During investigation:
    #     dry_run=True
    #
    # Only after human approval:
    #     dry_run=False
    # -----------------------------------------------------

    execution_arguments[
        "dry_run"
    ] = False

    print(
        "\n========== EXECUTING APPROVED ACTION =========="
    )

    print(
        f"Action: {proposed_action}"
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
            arguments=execution_arguments,
        )
    )

    result = parse_mcp_result(
        mcp_result
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
# Main Incident Agent
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
   if sufficient evidence supports the action.

You have access to diagnostic tools, anomaly-detection tools,
knowledge-search tools, and remediation tools through MCP.

IMPORTANT INVESTIGATION RULES:

1. Do not invent production facts.

2. Use diagnostic tools to collect current operational evidence.

3. Historical incidents and runbooks are supporting evidence.
   They do not prove the current root cause.

4. Prefer current production evidence when current evidence
   conflicts with historical knowledge.

5. Do not call tools unnecessarily.

6. Use search_knowledge when previous incidents or approved
   runbooks can help the investigation.

7. Remediation tools MUST ONLY be used in DRY-RUN mode
   during investigation.

8. Never execute remediation directly.

9. If an appropriate remediation tool exists and the root
   cause is sufficiently supported, call the remediation
   tool in dry-run mode.

10. Prefer the least risky remediation capable of restoring
    the system safely.

11. Do not consider a dry-run remediation to be executed.

When finished, provide:

1. Root Cause
2. Current Evidence
3. Relevant Historical Knowledge
4. Impact
5. Recommended Remediation
6. Validation Steps
"""

    input_items = [
        {
            "role": "user",
            "content": (
                "Investigate this production incident:\n\n"
                + json.dumps(
                    incident,
                    indent=2,
                )
            ),
        }
    ]

    proposed_remediation = None

    while True:

        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=llm_tools,
            input=input_items,
        )

        input_items.extend(
            response.output
        )

        tool_called = False

        for item in response.output:

            if item.type != "function_call":
                continue

            tool_called = True

            arguments = json.loads(
                item.arguments
            )

            print(
                "\n--------------------------------"
            )

            print(
                f"Agent selected MCP tool: "
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
            # Hard remediation guardrail
            # =================================================

            if item.name in REMEDIATION_TOOLS:

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
            # Execute selected tool through MCP
            # -------------------------------------------------

            mcp_result = await (
                mcp_session.call_tool(
                    item.name,
                    arguments=arguments,
                )
            )

            result = parse_mcp_result(
                mcp_result
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
            # Capture remediation proposal
            # -------------------------------------------------

            if item.name in REMEDIATION_TOOLS:

                proposed_remediation = {
                    "action":
                        item.name,

                    "arguments":
                        arguments,

                    "dry_run_result":
                        result,
                }

            # -------------------------------------------------
            # Give tool result back to LLM
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

        # -----------------------------------------------------
        # Investigation complete
        # -----------------------------------------------------

        if not tool_called:

            return {
                "final_rca":
                    response.output_text,

                "root_cause_summary":
                    response.output_text,

                "proposed_remediation":
                    proposed_remediation,
            }


# =========================================================
# Main
# =========================================================

async def main():

    incidents = load_incidents()

    # -----------------------------------------------------
    # Incident examples
    #
    # incidents[0] -> schema-change incident
    # incidents[1] -> duplicate/retry incident
    # incidents[2] -> anomaly incident
    #
    # We use duplicate/retry for remediation demo.
    # -----------------------------------------------------

    incident = incidents[1]

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

    server_params = StdioServerParameters(
        command=sys.executable,

        args=[
            "mcp_server.py",
        ],

        env=dict(
            os.environ
        ),
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
            # MCP handshake
            # =============================================

            await session.initialize()

            # =============================================
            # MCP dynamic tool discovery
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

            # =============================================
            # Convert MCP tools for LLM
            # =============================================

            llm_tools = (
                convert_mcp_tools_to_openai(
                    tools_response.tools
                )
            )

            # =============================================
            # Agent investigation
            # =============================================

            investigation = (
                await investigate_incident(
                    incident=incident,
                    mcp_session=session,
                    llm_tools=llm_tools,
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

            # =============================================
            # Check remediation proposal
            # =============================================

            proposed = investigation[
                "proposed_remediation"
            ]

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
            # Human approval
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
                    {}
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
            # Human rejected remediation
            # =============================================

            if approval_status != "APPROVED":

                print(
                    "\nRemediation was not approved. "
                    "No production action will be executed."
                )

                return

            print(
                "\nHuman approved remediation."
            )

            # =============================================
            # Execute approved remediation through MCP
            # =============================================

            execution_result = (
                await execute_approved_remediation(
                    mcp_session=session,

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
            # Stop if execution itself failed
            # =============================================

            if (
                not isinstance(
                    execution_result,
                    dict
                )
                or execution_result.get(
                    "status"
                ) != "EXECUTED"
            ):

                print(
                    "\nRemediation execution did not "
                    "complete successfully."
                )

                print(
                    "Incident remains OPEN."
                )

                return

            # =============================================
            # Post-remediation validation
            #
            # In a real environment the remediation tool
            # would return the new run_id.
            #
            # For our simulated scenario we use the
            # predefined recovery run.
            # =============================================

            recovery_run_id = (
                "RUN-1002-RECOVERY"
            )

            validation_result = (
                await validate_after_remediation(
                    mcp_session=session,

                    recovery_run_id=
                        recovery_run_id,

                    expected_records=
                        10000000,

                    duplicate_tolerance=
                        0,
                )
            )

            # =============================================
            # Determine final incident status
            # =============================================

            if (
                isinstance(
                    validation_result,
                    dict
                )
                and validation_result.get(
                    "validation_passed"
                )
            ):

                print(
                    "\n========================================"
                )

                print(
                    "INCIDENT RESOLVED"
                )

                print(
                    "Remediation executed and all "
                    "validation checks passed."
                )

                print(
                    "========================================"
                )

            else:

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
                    "Additional investigation is required."
                )

                print(
                    "========================================"
                )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
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
    "gpt-5.6"
)


# =========================================================
# Load incidents
# =========================================================

def load_incidents():
    with open("data/incidents.json") as f:
        return json.load(f)


# =========================================================
# Convert MCP tools to LLM tool definitions
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
# Convert MCP result into normal Python data
# =========================================================

def parse_mcp_result(result):

    if getattr(
        result,
        "structuredContent",
        None
    ) is not None:
        return result.structuredContent

    if getattr(
        result,
        "structured_content",
        None
    ) is not None:
        return result.structured_content

    texts = []

    for item in result.content:

        if getattr(
            item,
            "text",
            None
        ):
            texts.append(
                item.text
            )

    if not texts:
        return None

    combined = "\n".join(texts)

    try:
        return json.loads(
            combined
        )

    except json.JSONDecodeError:
        return combined


# =========================================================
# Remediation state
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
    state: RemediationState
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
        in [
            "yes",
            "y",
            "approve",
            "approved",
        ]
    )

    return {
        "approved": approved
    }


# =========================================================
# Approval routing
# =========================================================

def approval_router(
    state: RemediationState
):

    if state["approved"]:
        return "execute"

    return "rejected"


# =========================================================
# Approved execution node
# =========================================================

def execute_remediation_node(
    state: RemediationState
):

    print(
        "\nExecuting approved remediation..."
    )

    action = state[
        "proposed_action"
    ]

    arguments = state[
        "action_arguments"
    ]

    # -----------------------------------------------------
    # IMPORTANT
    #
    # Execution is still simulated here.
    #
    # In the next step we will call the real MCP action
    # tool after approval.
    # -----------------------------------------------------

    result = {
        "status":
            "EXECUTED",

        "action":
            action,

        "arguments":
            arguments,

        "message":
            (
                f"Approved remediation "
                f"{action} executed successfully."
            ),
    }

    return {
        "execution_result": result
    }


# =========================================================
# Rejected remediation node
# =========================================================

def rejected_node(
    state: RemediationState
):

    return {
        "execution_result": {
            "status":
                "REJECTED",

            "action":
                state[
                    "proposed_action"
                ],

            "message":
                (
                    "Human reviewer rejected "
                    "the proposed remediation."
                ),
        }
    }


# =========================================================
# Build LangGraph
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
        "execute",
        execute_remediation_node,
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
            "execute":
                "execute",

            "rejected":
                "rejected",
        },
    )

    builder.add_edge(
        "execute",
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
# Run human approval workflow
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
    # Run until interrupt()
    # -----------------------------------------------------

    result = graph.invoke(
        state,
        config=config,
    )

    interrupts = result.get(
        "__interrupt__",
        []
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
            pending
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
    # Resume SAME LangGraph thread
    # -----------------------------------------------------

    result = graph.invoke(
        Command(
            resume=decision
        ),
        config=config,
    )

    return result


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

Your objective is to:

1. investigate the incident,
2. identify the root cause,
3. search historical incidents and runbooks when useful,
4. determine downstream impact,
5. propose the safest remediation,
6. call an appropriate remediation tool in DRY-RUN mode
   when sufficient evidence supports that remediation.

You have access to diagnostic, anomaly-detection,
knowledge-search, and remediation tools exposed through MCP.

IMPORTANT RULES:

1. Do not invent production facts.

2. Use diagnostic tools to gather current operational evidence.

3. Historical incidents are supporting evidence only.
   They do not prove the current root cause.

4. Prefer current production evidence when it conflicts
   with historical knowledge.

5. Do not call tools unnecessarily.

6. Use search_knowledge when historical incidents or
   approved runbooks can help.

7. Remediation tools are DRY-RUN ONLY.

8. After identifying a sufficiently supported root cause,
   if one of the available remediation tools is appropriate,
   you MUST call that remediation tool in dry-run mode.

9. Do not treat the dry-run result as actual execution.

10. Prefer the least risky remediation action.

11. Do not call a remediation tool if none of the available
    tools safely matches the required remediation.

When the investigation is complete, provide:

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
                    "Investigate this production incident:\n\n"
                    + json.dumps(
                        incident,
                        indent=2
                    )
                )
        }
    ]

    proposed_remediation = None

    root_cause_summary = None

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
                "\nArguments:"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2
                )
            )

            # -------------------------------------------------
            # Execute tool through MCP
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
                    default=str
                )
            )

            # -------------------------------------------------
            # Capture remediation proposal
            # -------------------------------------------------

            if item.name in [
                "rerun_pipeline",
                "quarantine_partition",
                "rollback_deployment",
            ]:

                proposed_remediation = {
                    "action":
                        item.name,

                    "arguments":
                        arguments,

                    "dry_run_result":
                        result,
                }

            input_items.append(
                {
                    "type":
                        "function_call_output",

                    "call_id":
                        item.call_id,

                    "output":
                        json.dumps(
                            result,
                            default=str
                        ),
                }
            )

        # -----------------------------------------------------
        # No more tools requested
        # -----------------------------------------------------

        if not tool_called:

            root_cause_summary = (
                response.output_text
            )

            return {
                "final_rca":
                    response.output_text,

                "root_cause_summary":
                    root_cause_summary,

                "proposed_remediation":
                    proposed_remediation,
            }


# =========================================================
# Main
# =========================================================

async def main():

    incidents = load_incidents()

    # -----------------------------------------------------
    # Choose incident
    #
    # incidents[0] -> schema failure
    # incidents[1] -> duplicate retry issue
    # incidents[2] -> anomaly scenario
    # -----------------------------------------------------

    incident = incidents[1]

    print(
        "\n========== INCIDENT =========="
    )

    print(
        json.dumps(
            incident,
            indent=2
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
            write
        ) as session:

            # -------------------------------------------------
            # MCP handshake
            # -------------------------------------------------

            await session.initialize()

            # -------------------------------------------------
            # MCP dynamic tool discovery
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Convert MCP tools for LLM
            # -------------------------------------------------

            llm_tools = (
                convert_mcp_tools_to_openai(
                    tools_response.tools
                )
            )

            # -------------------------------------------------
            # Run incident investigation
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Check whether agent proposed remediation
            # -------------------------------------------------

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
                    default=str
                )
            )

            # -------------------------------------------------
            # Human-in-the-loop approval
            # -------------------------------------------------

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

            print(
                "\n========== REMEDIATION RESULT =========="
            )

            print(
                json.dumps(
                    approval_result.get(
                        "execution_result"
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
        main()
    )
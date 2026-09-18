import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()

client = OpenAI()

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6"
)


def load_incidents():
    with open("data/incidents.json") as f:
        return json.load(f)


# ---------------------------------------------------------
# Convert MCP tools into tools understood by the LLM
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Convert MCP tool output into normal Python data
# ---------------------------------------------------------

def parse_mcp_result(result):

    # Newer MCP tools may provide structured output directly.
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent

    # Fall back to text content.
    texts = []

    for item in result.content:

        if getattr(item, "text", None):
            texts.append(item.text)

    if not texts:
        return None

    combined = "\n".join(texts)

    # Some tools return JSON serialized as text.
    try:
        return json.loads(combined)
    except json.JSONDecodeError:
        return combined


# ---------------------------------------------------------
# Main Incident Agent
# ---------------------------------------------------------

async def investigate_incident(
    incident,
    mcp_session,
    llm_tools,
):

    instructions = """
You are a production data engineering incident investigator.

Your objective is to determine the root cause of a
production data incident and recommend a safe recovery plan.

You have access to tools exposed through an MCP server.

Use diagnostic tools to retrieve current operational evidence.
Use the knowledge-search tool when historical incidents or
approved runbooks can help with diagnosis or recovery.

Rules:

1. Do not invent production facts.

2. Gather current evidence using available tools.

3. Historical incidents are supporting evidence only.
   They do not prove the current root cause.

4. Prefer current production evidence if it conflicts
   with historical information.

5. Do not call tools unnecessarily.

6. Use lineage or other impact information when needed
   to determine downstream effects.

7. Do not recommend destructive actions unless they are
   justified by evidence.

When you have enough evidence, return:

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
                    indent=2
                )
            )
        }
    ]

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
                "Arguments:"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2
                )
            )

            # ---------------------------------------------
            # IMPORTANT:
            # Tool execution now happens through MCP.
            # ---------------------------------------------

            mcp_result = await mcp_session.call_tool(
                item.name,
                arguments=arguments,
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

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        result,
                        default=str
                    ),
                }
            )

        # Model did not request another tool.
        # Investigation is complete.
        if not tool_called:
            return response.output_text


# ---------------------------------------------------------
# MCP Client
# ---------------------------------------------------------

async def main():

    incidents = load_incidents()

    # Start with the schema incident.
    incident = incidents[0]

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
    # Tell MCP client how to launch our local server
    # -----------------------------------------------------

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "mcp_server.py",
        ],
        env=dict(os.environ),
    )

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            # MCP connection handshake
            await session.initialize()

            # ---------------------------------------------
            # MCP TOOL DISCOVERY
            # ---------------------------------------------

            tools_response = await session.list_tools()

            print(
                "\n========== MCP TOOLS =========="
            )

            for tool in tools_response.tools:
                print(
                    f"- {tool.name}"
                )

            # Translate discovered MCP tools into tool
            # definitions that the LLM can understand.
            llm_tools = convert_mcp_tools_to_openai(
                tools_response.tools
            )

            # Run investigation
            result = await investigate_incident(
                incident=incident,
                mcp_session=session,
                llm_tools=llm_tools,
            )

            print(
                "\n========== FINAL RCA =========="
            )

            print(result)


if __name__ == "__main__":
    asyncio.run(main())
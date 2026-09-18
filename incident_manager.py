import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from diagnostic_tools import (
    get_job_logs,
    get_schema_diff,
    get_job_metrics,
    get_recent_deployments,
    get_lineage,
)


load_dotenv()

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")


def load_incidents():
    with open("data/incidents.json") as f:
        return json.load(f)


# Tools exposed to the LLM
TOOLS = [
    {
        "type": "function",
        "name": "get_job_logs",
        "description": "Get error logs for a pipeline run.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string"
                }
            },
            "required": ["run_id"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_schema_diff",
        "description": "Compare the previous and current schema of a pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "string"
                }
            },
            "required": ["pipeline"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_job_metrics",
        "description": "Get runtime and processing metrics for a pipeline run.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string"
                }
            },
            "required": ["run_id"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_recent_deployments",
        "description": "Get recent deployments related to a pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "string"
                }
            },
            "required": ["pipeline"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_lineage",
        "description": "Get upstream and downstream dependencies of a pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "pipeline": {
                    "type": "string"
                }
            },
            "required": ["pipeline"],
            "additionalProperties": False
        }
    }
]


def execute_tool(name, arguments):

    if name == "get_job_logs":
        return get_job_logs(arguments["run_id"])

    if name == "get_schema_diff":
        return get_schema_diff(arguments["pipeline"])

    if name == "get_job_metrics":
        return get_job_metrics(arguments["run_id"])

    if name == "get_recent_deployments":
        return get_recent_deployments(arguments["pipeline"])

    if name == "get_lineage":
        return get_lineage(arguments["pipeline"])

    raise ValueError(f"Unknown tool: {name}")


def investigate_incident(incident):

    instructions = """
You are a production data engineering incident investigator.

Your job is to determine the root cause of a pipeline incident.

Use the available diagnostic tools to gather evidence.

Do not invent production facts.
Use tools when evidence is required.

When you have enough evidence, provide:

1. Root cause
2. Evidence
3. Impact
4. Recommended remediation
"""

    input_items = [
        {
            "role": "user",
            "content": (
                "Investigate this production incident:\n"
                + json.dumps(incident, indent=2)
            )
        }
    ]

    while True:

        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=TOOLS,
            input=input_items
        )

        # Preserve model output for next iteration
        input_items.extend(response.output)

        tool_called = False

        for item in response.output:

            if item.type != "function_call":
                continue

            tool_called = True

            arguments = json.loads(item.arguments)

            print(
                f"\nLLM selected tool: "
                f"{item.name}({arguments})"
            )

            result = execute_tool(
                item.name,
                arguments
            )

            print("Tool result:")
            print(result)

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                }
            )

        # No more tools requested.
        # The model has produced its final RCA.
        if not tool_called:
            return response.output_text


if __name__ == "__main__":

    incidents = load_incidents()

    # Start only with INC-1001
    incident = incidents[0]

    print("\nStarting investigation...")
    print(json.dumps(incident, indent=2))

    result = investigate_incident(incident)

    print("\n========== INCIDENT RCA ==========")
    print(result)
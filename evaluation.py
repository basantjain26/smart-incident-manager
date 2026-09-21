import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mcp import (
    ClientSession,
    StdioServerParameters,
)

from mcp.client.stdio import (
    stdio_client,
)

from incident_manager import (
    load_incidents,
    convert_mcp_tools_to_openai,
    investigate_incident,
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

EVALUATION_FILE = Path(
    "data/evaluation_cases.json"
)


# =========================================================
# Configurable cost rates
#
# Dollar cost per 1 million tokens.
#
# Keep pricing outside code because provider/model pricing
# can change.
# =========================================================

INPUT_COST_PER_1M = float(
    os.getenv(
        "EVAL_INPUT_COST_PER_1M",
        "0",
    )
)

OUTPUT_COST_PER_1M = float(
    os.getenv(
        "EVAL_OUTPUT_COST_PER_1M",
        "0",
    )
)


# =========================================================
# Load evaluation cases
# =========================================================

def load_evaluation_cases():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =========================================================
# Find incident
# =========================================================

def find_incident(
    incidents,
    incident_id,
):

    for incident in incidents:

        if (
            incident.get(
                "incident_id"
            )
            == incident_id
        ):

            return incident

    return None


# =========================================================
# Tool metrics
# =========================================================

def calculate_tool_metrics(
    expected_tools,
    allowed_extra_tools,
    tool_trace,
):

    actual_tools = [
        item[
            "tool"
        ]
        for item in tool_trace
    ]

    actual_unique = list(
        dict.fromkeys(
            actual_tools
        )
    )

    expected_set = set(
        expected_tools
    )

    allowed_set = set(
        allowed_extra_tools
    )

    actual_set = set(
        actual_unique
    )

    matched_expected = (
        expected_set
        & actual_set
    )

    if expected_set:

        recall = (
            len(
                matched_expected
            )
            / len(
                expected_set
            )
        )

    else:

        recall = 1.0

    acceptable_set = (
        expected_set
        | allowed_set
    )

    if actual_set:

        acceptable_called = (
            actual_set
            & acceptable_set
        )

        precision = (
            len(
                acceptable_called
            )
            / len(
                actual_set
            )
        )

    else:

        precision = (
            1.0
            if not expected_set
            else 0.0
        )

    return {
        "recall":
            recall,

        "precision":
            precision,

        "actual_tools":
            actual_unique,

        "missing_tools":
            sorted(
                expected_set
                - actual_set
            ),

        "unexpected_tools":
            sorted(
                actual_set
                - acceptable_set
            ),
    }


# =========================================================
# RAG recall
# =========================================================

def calculate_rag_recall(
    expected_documents,
    retrieved_documents,
):

    expected = set(
        expected_documents
    )

    retrieved = set(
        retrieved_documents
    )

    if not expected:

        return {
            "score":
                1.0,

            "matched":
                [],

            "missing":
                [],
        }

    matched = (
        expected
        & retrieved
    )

    missing = (
        expected
        - retrieved
    )

    return {
        "score":
            (
                len(matched)
                / len(expected)
            ),

        "matched":
            sorted(
                matched
            ),

        "missing":
            sorted(
                missing
            ),
    }


# =========================================================
# RCA concept score
# =========================================================

def calculate_rca_score(
    keywords,
    rca_text,
):

    normalized = (
        rca_text.lower()
    )

    if not keywords:

        return {
            "score":
                1.0,

            "matched":
                [],

            "missing":
                [],
        }

    matched = []
    missing = []

    for keyword in keywords:

        if (
            keyword.lower()
            in normalized
        ):

            matched.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )

    return {
        "score":
            (
                len(matched)
                / len(keywords)
            ),

        "matched":
            matched,

        "missing":
            missing,
    }


# =========================================================
# Remediation score
# =========================================================

def calculate_remediation_score(
    expected_remediation,
    remediation_required,
    proposed_remediation,
):

    actual_action = None

    if proposed_remediation:

        actual_action = (
            proposed_remediation.get(
                "action"
            )
        )

    if remediation_required:

        passed = (
            actual_action
            == expected_remediation
        )

    else:

        passed = True

    return {
        "score":
            1.0 if passed else 0.0,

        "expected":
            expected_remediation,

        "actual":
            actual_action,

        "passed":
            passed,
    }


# =========================================================
# Safety score
# =========================================================

def calculate_safety_score(
    safety_violations,
):

    passed = (
        len(
            safety_violations
        )
        == 0
    )

    return {
        "score":
            1.0 if passed else 0.0,

        "passed":
            passed,

        "violations":
            safety_violations,
    }


# =========================================================
# Estimated cost
# =========================================================

def calculate_estimated_cost(
    input_tokens,
    output_tokens,
):

    input_cost = (
        input_tokens
        / 1_000_000
    ) * INPUT_COST_PER_1M

    output_cost = (
        output_tokens
        / 1_000_000
    ) * OUTPUT_COST_PER_1M

    total_cost = (
        input_cost
        + output_cost
    )

    return {
        "input_cost":
            input_cost,

        "output_cost":
            output_cost,

        "total_cost":
            total_cost,
    }


# =========================================================
# Score one evaluation case
# =========================================================

def score_case(
    evaluation_case,
    investigation,
):

    tool_metrics = (
        calculate_tool_metrics(
            expected_tools=
                evaluation_case.get(
                    "expected_tools",
                    [],
                ),

            allowed_extra_tools=
                evaluation_case.get(
                    "allowed_extra_tools",
                    [],
                ),

            tool_trace=
                investigation.get(
                    "tool_trace",
                    [],
                ),
        )
    )

    rag_metrics = (
        calculate_rag_recall(
            expected_documents=
                evaluation_case.get(
                    "expected_rag_documents",
                    [],
                ),

            retrieved_documents=
                investigation.get(
                    "retrieved_documents",
                    [],
                ),
        )
    )

    rca_metrics = (
        calculate_rca_score(
            keywords=
                evaluation_case.get(
                    "root_cause_keywords",
                    [],
                ),

            rca_text=
                investigation.get(
                    "final_rca",
                    "",
                ),
        )
    )

    remediation_metrics = (
        calculate_remediation_score(
            expected_remediation=
                evaluation_case.get(
                    "expected_remediation"
                ),

            remediation_required=
                evaluation_case.get(
                    "remediation_required",
                    False,
                ),

            proposed_remediation=
                investigation.get(
                    "proposed_remediation"
                ),
        )
    )

    safety_metrics = (
        calculate_safety_score(
            safety_violations=
                investigation.get(
                    "safety_violations",
                    [],
                )
        )
    )

    telemetry = (
        investigation.get(
            "telemetry",
            {},
        )
    )

    input_tokens = (
        telemetry.get(
            "input_tokens",
            0,
        )
    )

    output_tokens = (
        telemetry.get(
            "output_tokens",
            0,
        )
    )

    estimated_cost = (
        calculate_estimated_cost(
            input_tokens=
                input_tokens,

            output_tokens=
                output_tokens,
        )
    )

    component_scores = [
        tool_metrics[
            "recall"
        ],

        tool_metrics[
            "precision"
        ],

        rag_metrics[
            "score"
        ],

        rca_metrics[
            "score"
        ],

        remediation_metrics[
            "score"
        ],

        safety_metrics[
            "score"
        ],
    ]

    overall_score = (
        sum(
            component_scores
        )
        / len(
            component_scores
        )
    )

    return {
        "case_id":
            evaluation_case[
                "case_id"
            ],

        "incident_id":
            evaluation_case[
                "incident_id"
            ],

        "description":
            evaluation_case.get(
                "description"
            ),

        # ---------------------------------------------
        # Quality
        # ---------------------------------------------

        "tool_recall":
            tool_metrics[
                "recall"
            ],

        "tool_precision":
            tool_metrics[
                "precision"
            ],

        "rag_recall":
            rag_metrics[
                "score"
            ],

        "rca_score":
            rca_metrics[
                "score"
            ],

        "remediation_score":
            remediation_metrics[
                "score"
            ],

        "safety_score":
            safety_metrics[
                "score"
            ],

        "overall_score":
            overall_score,

        # ---------------------------------------------
        # Operational telemetry
        # ---------------------------------------------

        "model":
            telemetry.get(
                "model"
            ),

        "llm_calls":
            telemetry.get(
                "llm_calls",
                0,
            ),

        "mcp_tool_calls":
            telemetry.get(
                "mcp_tool_calls",
                0,
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            telemetry.get(
                "total_tokens",
                0,
            ),

        "investigation_latency_seconds":
            telemetry.get(
                "investigation_latency_seconds",
                0.0,
            ),

        "llm_latency_seconds":
            telemetry.get(
                "llm_latency_seconds",
                0.0,
            ),

        "tool_latency_seconds":
            telemetry.get(
                "tool_latency_seconds",
                0.0,
            ),

        "estimated_input_cost":
            estimated_cost[
                "input_cost"
            ],

        "estimated_output_cost":
            estimated_cost[
                "output_cost"
            ],

        "estimated_cost":
            estimated_cost[
                "total_cost"
            ],

        "details": {
            "tools":
                tool_metrics,

            "rag":
                rag_metrics,

            "rca":
                rca_metrics,

            "remediation":
                remediation_metrics,

            "safety":
                safety_metrics,
        },
    }


# =========================================================
# Formatting
# =========================================================

def percentage(
    value,
):

    return (
        f"{value * 100:.1f}%"
    )


# =========================================================
# Print per-case result
# =========================================================

def print_case_result(
    result,
):

    print(
        "\n========================================"
    )

    print(
        f"{result['case_id']} "
        f"- {result['incident_id']}"
    )

    print(
        result[
            "description"
        ]
    )

    print(
        "----------------------------------------"
    )

    print(
        "Tool Recall:          ",
        percentage(
            result[
                "tool_recall"
            ]
        ),
    )

    print(
        "Tool Precision:       ",
        percentage(
            result[
                "tool_precision"
            ]
        ),
    )

    print(
        "RAG Recall:           ",
        percentage(
            result[
                "rag_recall"
            ]
        ),
    )

    print(
        "RCA Concept Score:    ",
        percentage(
            result[
                "rca_score"
            ]
        ),
    )

    print(
        "Remediation Accuracy: ",
        percentage(
            result[
                "remediation_score"
            ]
        ),
    )

    print(
        "Safety Compliance:    ",
        percentage(
            result[
                "safety_score"
            ]
        ),
    )

    print(
        "----------------------------------------"
    )

    print(
        "Overall Score:        ",
        percentage(
            result[
                "overall_score"
            ]
        ),
    )

    print(
        "\nOperational Metrics"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Model:                 ",
        result[
            "model"
        ],
    )

    print(
        "LLM Calls:             ",
        result[
            "llm_calls"
        ],
    )

    print(
        "MCP Tool Calls:        ",
        result[
            "mcp_tool_calls"
        ],
    )

    print(
        "Input Tokens:          ",
        result[
            "input_tokens"
        ],
    )

    print(
        "Output Tokens:         ",
        result[
            "output_tokens"
        ],
    )

    print(
        "Total Tokens:          ",
        result[
            "total_tokens"
        ],
    )

    print(
        "Investigation Latency: ",
        (
            f"{result['investigation_latency_seconds']:.2f}s"
        ),
    )

    print(
        "LLM Latency:           ",
        (
            f"{result['llm_latency_seconds']:.2f}s"
        ),
    )

    print(
        "Tool Latency:          ",
        (
            f"{result['tool_latency_seconds']:.2f}s"
        ),
    )

    print(
        "Estimated LLM Cost:    ",
        (
            f"${result['estimated_cost']:.6f}"
        ),
    )

    print(
        "========================================"
    )


# =========================================================
# Aggregate quality metrics
# =========================================================

def aggregate_results(
    results,
):

    if not results:
        return {}

    fields = [
        "tool_recall",
        "tool_precision",
        "rag_recall",
        "rca_score",
        "remediation_score",
        "safety_score",
        "overall_score",
    ]

    summary = {}

    for field in fields:

        summary[
            field
        ] = (
            sum(
                result[
                    field
                ]
                for result in results
            )
            / len(results)
        )

    return summary


# =========================================================
# Aggregate operational metrics
# =========================================================

def aggregate_operational_metrics(
    results,
):

    if not results:

        return {
            "evaluation_cases":
                0,

            "total_llm_calls":
                0,

            "total_mcp_tool_calls":
                0,

            "total_input_tokens":
                0,

            "total_output_tokens":
                0,

            "total_tokens":
                0,

            "average_latency_seconds":
                0.0,

            "average_llm_latency_seconds":
                0.0,

            "average_tool_latency_seconds":
                0.0,

            "estimated_total_cost":
                0.0,
        }

    count = len(
        results
    )

    return {
        "evaluation_cases":
            count,

        "total_llm_calls":
            sum(
                result[
                    "llm_calls"
                ]
                for result in results
            ),

        "total_mcp_tool_calls":
            sum(
                result[
                    "mcp_tool_calls"
                ]
                for result in results
            ),

        "total_input_tokens":
            sum(
                result[
                    "input_tokens"
                ]
                for result in results
            ),

        "total_output_tokens":
            sum(
                result[
                    "output_tokens"
                ]
                for result in results
            ),

        "total_tokens":
            sum(
                result[
                    "total_tokens"
                ]
                for result in results
            ),

        "average_latency_seconds":
            (
                sum(
                    result[
                        "investigation_latency_seconds"
                    ]
                    for result in results
                )
                / count
            ),

        "average_llm_latency_seconds":
            (
                sum(
                    result[
                        "llm_latency_seconds"
                    ]
                    for result in results
                )
                / count
            ),

        "average_tool_latency_seconds":
            (
                sum(
                    result[
                        "tool_latency_seconds"
                    ]
                    for result in results
                )
                / count
            ),

        "estimated_total_cost":
            sum(
                result[
                    "estimated_cost"
                ]
                for result in results
            ),
    }


# =========================================================
# Run evaluation suite
# =========================================================

async def run_evaluations():

    incidents = (
        load_incidents()
    )

    evaluation_cases = (
        load_evaluation_cases()
    )

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

    results = []

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

            tools_response = (
                await session.list_tools()
            )

            llm_tools = (
                convert_mcp_tools_to_openai(
                    tools_response.tools
                )
            )

            for evaluation_case in (
                evaluation_cases
            ):

                print(
                    "\n\n########################################"
                )

                print(
                    "RUNNING "
                    f"{evaluation_case['case_id']}"
                )

                print(
                    "########################################"
                )

                incident = (
                    find_incident(
                        incidents,
                        evaluation_case[
                            "incident_id"
                        ],
                    )
                )

                if incident is None:

                    print(
                        "Incident not found: "
                        f"{evaluation_case['incident_id']}"
                    )

                    continue

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

                result = (
                    score_case(
                        evaluation_case,
                        investigation,
                    )
                )

                results.append(
                    result
                )

                print_case_result(
                    result
                )

    # =====================================================
    # Aggregate quality summary
    # =====================================================

    quality_summary = (
        aggregate_results(
            results
        )
    )

    print(
        "\n\n========================================"
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "========================================"
    )

    if quality_summary:

        print(
            "Tool Recall:          ",
            percentage(
                quality_summary[
                    "tool_recall"
                ]
            ),
        )

        print(
            "Tool Precision:       ",
            percentage(
                quality_summary[
                    "tool_precision"
                ]
            ),
        )

        print(
            "RAG Recall:           ",
            percentage(
                quality_summary[
                    "rag_recall"
                ]
            ),
        )

        print(
            "RCA Concept Score:    ",
            percentage(
                quality_summary[
                    "rca_score"
                ]
            ),
        )

        print(
            "Remediation Accuracy: ",
            percentage(
                quality_summary[
                    "remediation_score"
                ]
            ),
        )

        print(
            "Safety Compliance:    ",
            percentage(
                quality_summary[
                    "safety_score"
                ]
            ),
        )

        print(
            "----------------------------------------"
        )

        print(
            "Overall Score:        ",
            percentage(
                quality_summary[
                    "overall_score"
                ]
            ),
        )

    # =====================================================
    # Aggregate operational summary
    # =====================================================

    operational_summary = (
        aggregate_operational_metrics(
            results
        )
    )

    print(
        "\n========================================"
    )

    print(
        "OPERATIONAL SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Evaluation Cases:      ",
        operational_summary[
            "evaluation_cases"
        ],
    )

    print(
        "Total LLM Calls:       ",
        operational_summary[
            "total_llm_calls"
        ],
    )

    print(
        "Total MCP Tool Calls:  ",
        operational_summary[
            "total_mcp_tool_calls"
        ],
    )

    print(
        "Total Input Tokens:    ",
        operational_summary[
            "total_input_tokens"
        ],
    )

    print(
        "Total Output Tokens:   ",
        operational_summary[
            "total_output_tokens"
        ],
    )

    print(
        "Total Tokens:          ",
        operational_summary[
            "total_tokens"
        ],
    )

    print(
        "Average Latency:       ",
        (
            f"{operational_summary['average_latency_seconds']:.2f}s"
        ),
    )

    print(
        "Average LLM Latency:   ",
        (
            f"{operational_summary['average_llm_latency_seconds']:.2f}s"
        ),
    )

    print(
        "Average Tool Latency:  ",
        (
            f"{operational_summary['average_tool_latency_seconds']:.2f}s"
        ),
    )

    print(
        "Estimated Total Cost:  ",
        (
            f"${operational_summary['estimated_total_cost']:.6f}"
        ),
    )

    print(
        "========================================"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        run_evaluations()
    )
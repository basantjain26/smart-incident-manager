import json


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_job_logs(run_id):
    data = load_json("data/job_logs.json")
    return data.get(run_id)


def get_schema_diff(pipeline):
    data = load_json("data/schemas.json")
    schema = data.get(pipeline)

    if not schema:
        return None

    previous = set(schema["previous"])
    current = set(schema["current"])

    return {
        "removed_columns": list(previous - current),
        "added_columns": list(current - previous)
    }


def get_job_metrics(run_id):
    data = load_json("data/job_metrics.json")
    return data.get(run_id)


def get_recent_deployments(pipeline):
    data = load_json("data/deployments.json")
    return data.get(pipeline, [])


def get_lineage(pipeline):
    data = load_json("data/lineage.json")
    return data.get(pipeline)


if __name__ == "__main__":
    print("Logs:")
    print(get_job_logs("RUN-1001"))

    print("\nSchema Diff:")
    print(get_schema_diff("customer_transactions"))

    print("\nMetrics:")
    print(get_job_metrics("RUN-1001"))

    print("\nDeployments:")
    print(get_recent_deployments("customer_transactions"))

    print("\nLineage:")
    print(get_lineage("customer_transactions"))
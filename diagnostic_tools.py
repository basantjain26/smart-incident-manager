import json
from sklearn.ensemble import IsolationForest


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

def get_pipeline_metric_history(pipeline):
    data = load_json(
        "data/pipeline_metric_history.json"
    )

    return data.get(pipeline, [])


def detect_pipeline_anomaly(pipeline, run_id):

    history = get_pipeline_metric_history(
        pipeline
    )

    current = get_job_metrics(
        run_id
    )

    if not history or not current:
        return {
            "is_anomaly": False,
            "reason": "Insufficient metrics"
        }

    features = [
        "runtime_minutes",
        "records_written",
        "duplicate_rate",
        "freshness_delay_minutes",
    ]

    # ---------------------------------------------
    # Rule-based detection
    # ---------------------------------------------

    rule_alerts = []

    if current["runtime_minutes"] > 60:
        rule_alerts.append(
            "Runtime exceeded 60 minutes"
        )

    if current["duplicate_rate"] > 0.05:
        rule_alerts.append(
            "Duplicate rate exceeded 5%"
        )

    if current["freshness_delay_minutes"] > 30:
        rule_alerts.append(
            "Freshness delay exceeded 30 minutes"
        )

    # ---------------------------------------------
    # ML-based anomaly detection
    # ---------------------------------------------

    training_data = [
        [
            row[feature]
            for feature in features
        ]
        for row in history
    ]

    current_vector = [
        current[feature]
        for feature in features
    ]

    model = IsolationForest(
        contamination=0.1,
        random_state=42,
    )

    model.fit(training_data)

    prediction = model.predict(
        [current_vector]
    )[0]

    anomaly_score = model.decision_function(
        [current_vector]
    )[0]

    ml_anomaly = prediction == -1

    return {
        "pipeline": pipeline,
        "run_id": run_id,
        "rule_alerts": rule_alerts,
        "ml_anomaly": bool(ml_anomaly),
        "anomaly_score": round(
            float(anomaly_score),
            4
        ),
        "is_anomaly": (
            bool(rule_alerts)
            or bool(ml_anomaly)
        ),
        "current_metrics": current,
    }

def rerun_pipeline(
    pipeline,
    run_id,
    dry_run=True,
):
    action = {
        "action": "rerun_pipeline",
        "pipeline": pipeline,
        "run_id": run_id,
        "dry_run": dry_run,
    }

    if dry_run:
        action["status"] = "PROPOSED"
        action["message"] = (
            f"Would rerun pipeline {pipeline} "
            f"for failed run {run_id}."
        )

        return action

    # Real production implementation would call
    # Airflow / Databricks / Glue here.
    action["status"] = "EXECUTED"

    return action


def quarantine_partition(
    pipeline,
    partition="affected_batch",
    dry_run=True,
):
    action = {
        "action": "quarantine_partition",
        "pipeline": pipeline,
        "partition": partition,
        "dry_run": dry_run,
    }

    if dry_run:
        action["status"] = "PROPOSED"
        action["message"] = (
            f"Would quarantine partition "
            f"{partition} for pipeline {pipeline}."
        )

        return action

    action["status"] = "EXECUTED"

    return action


def rollback_deployment(
    pipeline,
    version,
    dry_run=True,
):
    action = {
        "action": "rollback_deployment",
        "pipeline": pipeline,
        "version": version,
        "dry_run": dry_run,
    }

    if dry_run:
        action["status"] = "PROPOSED"
        action["message"] = (
            f"Would roll back pipeline {pipeline} "
            f"to version {version}."
        )

        return action

    action["status"] = "EXECUTED"

    return action


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
    result = detect_pipeline_anomaly(
        pipeline="customer_gold",
        run_id="RUN-1003",
    )
    print(result)
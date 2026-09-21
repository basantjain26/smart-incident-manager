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
    if dry_run:
        return {
            "action": "rerun_pipeline",
            "status": "PROPOSED",
            "dry_run": True,
            "pipeline": pipeline,
            "run_id": run_id,
            "message": (
                f"Would rerun pipeline {pipeline} "
                f"for run {run_id}."
            ),
        }

    # Production implementation:
    # call Airflow / Databricks / Glue API here.

    return {
        "action": "rerun_pipeline",
        "status": "EXECUTED",
        "dry_run": False,
        "pipeline": pipeline,
        "run_id": run_id,
        "message": (
            f"Pipeline {pipeline} rerun executed."
        ),
    }


def quarantine_partition(
    pipeline,
    partition="affected_batch",
    dry_run=True,
):
    if dry_run:
        return {
            "action": "quarantine_partition",
            "status": "PROPOSED",
            "dry_run": True,
            "pipeline": pipeline,
            "partition": partition,
            "message": (
                f"Would quarantine {partition} "
                f"for pipeline {pipeline}."
            ),
        }

    # Production implementation:
    # move data to quarantine location / mark partition invalid.

    return {
        "action": "quarantine_partition",
        "status": "EXECUTED",
        "dry_run": False,
        "pipeline": pipeline,
        "partition": partition,
        "message": (
            f"Partition {partition} quarantined "
            f"for pipeline {pipeline}."
        ),
    }


def rollback_deployment(
    pipeline,
    version,
    dry_run=True,
):
    if dry_run:
        return {
            "action": "rollback_deployment",
            "status": "PROPOSED",
            "dry_run": True,
            "pipeline": pipeline,
            "version": version,
            "message": (
                f"Would roll back {pipeline} "
                f"to version {version}."
            ),
        }

    return {
        "action": "rollback_deployment",
        "status": "EXECUTED",
        "dry_run": False,
        "pipeline": pipeline,
        "version": version,
        "message": (
            f"Pipeline {pipeline} rolled back "
            f"to version {version}."
        ),
    }

def validate_remediation(
    recovery_run_id,
    expected_records=None,
    duplicate_tolerance=0,
):
    metrics = get_job_metrics(
        recovery_run_id
    )

    if not metrics:
        return {
            "validation_passed": False,
            "reason": "Recovery run metrics not found",
        }

    checks = {}

    # ---------------------------------------------
    # Pipeline status
    # ---------------------------------------------

    checks["pipeline_success"] = (
        metrics.get("status") == "SUCCESS"
    )

    # ---------------------------------------------
    # Duplicate validation
    # ---------------------------------------------

    duplicate_records = metrics.get(
        "duplicate_records",
        0
    )

    checks["duplicates_within_tolerance"] = (
        duplicate_records <= duplicate_tolerance
    )

    # ---------------------------------------------
    # Expected record count
    # ---------------------------------------------

    if expected_records is not None:

        actual_records = metrics.get(
            "records_written",
            0
        )

        tolerance = expected_records * 0.02

        checks["record_count_valid"] = (
            abs(
                actual_records - expected_records
            )
            <= tolerance
        )

    # ---------------------------------------------
    # Freshness check
    # ---------------------------------------------

    checks["freshness_ok"] = (
        metrics.get(
            "freshness_delay_minutes",
            9999
        )
        <= 30
    )

    validation_passed = all(
        checks.values()
    )

    return {
        "recovery_run_id":
            recovery_run_id,

        "validation_passed":
            validation_passed,

        "checks":
            checks,

        "metrics":
            metrics,
    }

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
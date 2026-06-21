"""Warmup 11: Prefect orchestration and production ETL patterns."""

from __future__ import annotations

from prefect import get_run_logger, task


# ============================================================
# Prefect Orchestration
# ============================================================

# Prefect Question 1
# @task wraps a unit of work so Prefect can orchestrate it as an observable step:
# retries, state tracking, caching, task run logs, and dependency edges are all
# attached at the task level. @flow wraps the full orchestration function that
# coordinates task calls, controls run-level behavior, and appears as the top-level
# pipeline run in the UI.
#
# For a helper that only converts Celsius to Fahrenheit (pure in-memory math, no
# network/file/database I/O), I usually would NOT decorate it with @task. Making it
# a normal Python function is simpler and avoids task-run overhead. I would only make
# it a task if I specifically needed task-level observability/retries/caching for it.


# Prefect Question 2
@task(name="call_api", retries=3, retry_delay_seconds=30)


# Prefect Question 3
# If extract is Completed, transform is Failed, and load never ran, I would open the
# failed flow run in the Prefect UI and inspect the task run graph/timeline, then
# click into the transform task run details.
#
# Specifically, I expect to find:
# - The transform task state and error type/message (stack trace/exception details)
# - Task run logs around the failure point (input context, retries, warnings)
# - Retry history and final terminal state after retries were exhausted
# - Timing metadata (start/end/duration) to see exactly when failure occurred
# - Upstream/downstream dependency view showing that load was skipped/not triggered


# ============================================================
# Production Patterns
# ============================================================

# Production Question 1
# raise_for_status() raises an HTTPError for 4xx/5xx responses, which causes the
# task to fail immediately and surfaces a real exception in orchestration logs/state.
# That is better than:
#   if response.status_code != 200: print("error")
# because printing does not fail the task by itself; the pipeline may continue with
# bad/empty data and hide the root cause.
#
# On a 500 response:
# - With raise_for_status(): the task fails, Prefect can retry/fail the run, and
#   downstream dependent tasks do not run.
# - With only print("error"): the task may still appear successful unless you
#   explicitly raise, so downstream tasks can run with invalid inputs.


# Production Question 2
# Writing to final/{today}/weather_etl.json with overwrite=True makes reruns
# idempotent for that target path: after fixing a crash and rerunning, the new
# successful output replaces any prior object at that same key.
#
# Without overwrite=True, upload may fail with a "blob already exists" conflict if
# something had been written previously for that key, requiring manual cleanup or a
# new path. overwrite=True protects reruns from that conflict and keeps one canonical
# artifact per run-date key.


# Production Question 3
@task(name="log-load-summary")
def log_load_summary(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info("Loaded %d records to %s", len(records), blob_path)

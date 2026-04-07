"""Prefect pipeline warmup exercise using Prefect tasks and flows"""

from prefect import flow, task
import numpy as np
import pandas as pd

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }

@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)

    print("Prefect pipeline result:")
    for key, value in summary.items():
        print(f"{key} = {value}")

    return summary

if __name__ == "__main__":
    pipeline_flow()

# Prefect overhead comment:
# This pipeline is simple and only processes a small array, so Prefect adds extra overhead
# in the form of task/flow orchestration, scheduling metadata, and runtime setup that is
# not necessary for plain Python function composition.
# Prefect is still useful when workflows need retries, logging, monitoring, parameterization,
# scheduling, or when tasks involve I/O, external systems, or multiple dependent steps.
# In those realistic cases, Prefect provides observability and reliability even if the core
# pipeline logic remains small.

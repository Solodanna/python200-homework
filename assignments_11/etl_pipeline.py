"""Week 11 ETL capstone: Extract from Open-Meteo, transform with OpenAI, load to Azure Blob."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task


ACCOUNT_URL = "https://annactd20266sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# City choice: Seattle, WA
LATITUDE = 47.6062
LONGITUDE = -122.3321

MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)


def _normalize_label(raw_label: str) -> str:
    cleaned = raw_label.strip().lower()
    return cleaned if cleaned in {"good", "marginal", "bad"} else "unknown"


@task(retries=2, retry_delay_seconds=10)
def extract_weather() -> dict[str, Any]:
    """Fetch 7 days of hourly weather data from Open-Meteo."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    print("Extract complete: received raw JSON from Open-Meteo.")
    return payload


@task
def transform_weather(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape hourly arrays to records and classify the first 24 with OpenAI."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    hourly = raw_payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precipitation = hourly.get("precipitation", [])

    count = min(len(times), len(temps), len(precipitation))
    records: list[dict[str, Any]] = []
    for idx in range(count):
        records.append(
            {
                "time": times[idx],
                "temperature_2m": float(temps[idx]),
                "precipitation": float(precipitation[idx]),
            }
        )

    client = OpenAI(api_key=api_key)
    classify_count = min(24, len(records))

    for idx in range(classify_count):
        record = records[idx]
        user_prompt = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        model_label = response.choices[0].message.content or ""
        records[idx]["conditions"] = _normalize_label(model_label)

        if (idx + 1) % 6 == 0:
            print(f"Transform progress: classified {idx + 1}/{classify_count} records.")

    for idx in range(classify_count, len(records)):
        records[idx]["conditions"] = "unknown"

    print(f"Transform complete: produced {len(records)} enriched records.")
    return records


@task
def load_weather(records: list[dict[str, Any]]) -> str:
    """Upload enriched weather records to Azure Blob Storage."""
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)

    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"
    payload = json.dumps(records, indent=2).encode("utf-8")

    container_client.upload_blob(name=blob_path, data=payload, overwrite=True)
    print(f"Load complete: uploaded {blob_path} ({len(payload)} bytes).")
    return blob_path


@flow(log_prints=True)
def weather_etl_flow() -> str:
    """Run extract, transform, and load tasks in order."""
    raw_payload = extract_weather()
    enriched_records = transform_weather(raw_payload)
    final_blob_path = load_weather(enriched_records)
    print(f"ETL flow complete. Final blob path: {final_blob_path}")
    return final_blob_path


if __name__ == "__main__":
    weather_etl_flow()

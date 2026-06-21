"""
Video URL: https://youtu.be/your-video-link-here
Project 10: LLM Transform pipeline on weather data from Blob Storage.

Classifying running conditions is a reasonable LLM use case when you want flexible,
language-aware judgment that can adapt to nuanced context and evolving criteria.
A deterministic rules engine could absolutely do this task and would be faster,
cheaper, and easier to test for strict consistency across large batches.
The tradeoff is that rule-based logic can be brittle and requires manual updates
whenever your definition of "good" conditions changes.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).parent
REPO_DIR = PROJECT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

ACCOUNT_URL = "https://annactd20266sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

FALLBACK_RAW_PATH = REPO_DIR / "assignments" / "resources" / "weather_raw.json"
FALLBACK_ALT_PATH = REPO_DIR / "assignments_09" / "outputs" / "weather_raw.json"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)


def _is_valid_api_key(api_key: str) -> bool:
    """Return True when API key is present and not a placeholder value."""
    if not api_key:
        return False
    lowered = api_key.lower()
    placeholders = ("your-key", "changeme", "replace-me", "placeholder")
    return not any(token in lowered for token in placeholders)


def get_blob_service_client() -> BlobServiceClient:
    """Create a BlobServiceClient using DefaultAzureCredential."""
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)


def reshape_hourly_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert parallel hourly arrays to per-hour record dictionaries."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])

    count = min(len(times), len(temps), len(precip))
    records: list[dict[str, Any]] = []
    for idx in range(count):
        records.append(
            {
                "time": times[idx],
                "temperature_2m": float(temps[idx]),
                "precipitation": float(precip[idx]),
            }
        )
    return records


def read_raw_weather(blob_service_client: BlobServiceClient) -> tuple[list[dict[str, Any]], str]:
    """Read raw weather from today's blob path or fall back to local dataset."""
    container_client = blob_service_client.get_container_client(CONTAINER)
    today = date.today().isoformat()
    raw_blob_path = f"raw/{today}/weather.json"

    try:
        raw_bytes = container_client.download_blob(raw_blob_path).readall()
        payload = json.loads(raw_bytes.decode("utf-8"))
        print(f"Loaded raw weather from Blob: {raw_blob_path}")
        return reshape_hourly_payload(payload), raw_blob_path
    except ResourceNotFoundError:
        print(f"Blob not found for today at {raw_blob_path}. Using fallback dataset.")

    for fallback_path in (FALLBACK_RAW_PATH, FALLBACK_ALT_PATH):
        if fallback_path.exists():
            with fallback_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            print(f"Loaded fallback weather from {fallback_path}")
            return reshape_hourly_payload(payload), str(fallback_path)

    raise FileNotFoundError(
        "No raw weather source found in Blob for today and no local fallback file exists."
    )


def _classify_record(client: OpenAI, record: dict[str, Any]) -> str:
    """Classify one record as good, marginal, or bad with unknown fallback."""
    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    label = (response.choices[0].message.content or "").strip().lower()
    return label if label in {"good", "marginal", "bad"} else "unknown"


def transform_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify the first 24 hourly records and append conditions."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not _is_valid_api_key(api_key):
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still a placeholder. "
            "Set a real key in .env before running project_10.py."
        )

    selected = records[:24]
    client = OpenAI(api_key=api_key)
    enriched: list[dict[str, Any]] = []

    for idx, record in enumerate(selected, start=1):
        conditions = _classify_record(client, record)
        enriched.append({**record, "conditions": conditions})

        if idx % 6 == 0:
            print(f"Processed {idx}/{len(selected)} records...")

    return enriched


def write_processed_blob(
    blob_service_client: BlobServiceClient, enriched_records: list[dict[str, Any]]
) -> str:
    """Upload enriched records to processed/<today>/weather_classified.json."""
    today = date.today().isoformat()
    processed_blob_path = f"processed/{today}/weather_classified.json"
    container_client = blob_service_client.get_container_client(CONTAINER)

    payload = json.dumps(enriched_records, indent=2).encode("utf-8")
    container_client.upload_blob(
        name=processed_blob_path,
        data=payload,
        overwrite=True,
    )
    print(f"Uploaded processed blob: {processed_blob_path}")
    return processed_blob_path


def spot_check(
    blob_service_client: BlobServiceClient, processed_blob_path: str
) -> pd.DataFrame:
    """Download processed blob, print value counts and first five rows."""
    container_client = blob_service_client.get_container_client(CONTAINER)
    blob_bytes = container_client.download_blob(processed_blob_path).readall()
    enriched_records = json.loads(blob_bytes.decode("utf-8"))

    df = pd.DataFrame(enriched_records)
    print(df["conditions"].value_counts())
    print(df.head())
    return df


def save_first_10(enriched_records: list[dict[str, Any]]) -> Path:
    """Save first 10 enriched records locally for mentor inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "first_10_records.json"
    output_path.write_text(json.dumps(enriched_records[:10], indent=2), encoding="utf-8")
    print(f"Saved first 10 records to {output_path}")
    return output_path


def main() -> None:
    """Run the Week 10 read -> transform -> write pipeline."""
    blob_service_client = get_blob_service_client()
    records, source = read_raw_weather(blob_service_client)
    print(f"Input source: {source}")

    enriched_records = transform_records(records)
    processed_blob_path = write_processed_blob(blob_service_client, enriched_records)
    spot_check(blob_service_client, processed_blob_path)
    save_first_10(enriched_records)


if __name__ == "__main__":
    main()

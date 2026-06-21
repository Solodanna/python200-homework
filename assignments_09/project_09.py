"""Project 09: Extract + Load pipeline using Open-Meteo and Azure Blob Storage.

Video URL: https://youtu.be/Na5DZvlO7bM
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

ACCOUNT_URL = "https://annactd20266sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
LATITUDE = 35.2271
LONGITUDE = -80.8431


def build_weather_url(latitude: float = LATITUDE, longitude: float = LONGITUDE) -> str:
    """Build the Open-Meteo API URL for 7 days of hourly weather data."""
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )


def get_credential() -> DefaultAzureCredential:
    """Return the Azure credential chain used throughout the assignment."""
    return DefaultAzureCredential(exclude_interactive_browser_credential=False)


def extract() -> bytes:
    """Fetch weather data from Open-Meteo and serialize it to UTF-8 JSON bytes."""
    response = requests.get(build_weather_url(), timeout=30)
    response.raise_for_status()

    payload = response.json()
    return json.dumps(payload).encode("utf-8")


def load(blob_service_client: BlobServiceClient, payload: bytes) -> str:
    """Upload the serialized weather payload to Blob Storage."""
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    container_client = blob_service_client.get_container_client(CONTAINER)
    container_client.upload_blob(name=blob_path, data=payload, overwrite=True)
    print(f"Uploaded {blob_path} ({len(payload)} bytes)")
    return blob_path


def list_blobs(blob_service_client: BlobServiceClient) -> None:
    """List all blobs in the container and print their names and sizes."""
    container_client = blob_service_client.get_container_client(CONTAINER)
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")


def read_back(blob_service_client: BlobServiceClient, blob_path: str) -> pd.DataFrame:
    """Download the uploaded blob, save it locally, and preview hourly data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    container_client = blob_service_client.get_container_client(CONTAINER)
    blob_bytes = container_client.download_blob(blob_path).readall()

    output_path = OUTPUT_DIR / "weather_raw.json"
    output_path.write_bytes(blob_bytes)

    payload = json.loads(blob_bytes.decode("utf-8"))
    frame = pd.DataFrame(payload["hourly"])
    print(frame.head())
    return frame


def main() -> None:
    """Run the Extract + Load pipeline."""
    credential = get_credential()
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)

    payload = extract()
    blob_path = load(blob_service_client, payload)
    list_blobs(blob_service_client)
    read_back(blob_service_client, blob_path)


if __name__ == "__main__":
    main()

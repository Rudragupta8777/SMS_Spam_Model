"""
Pulls field-reported samples from the telemetry backend into data/raw/telemetry_export.csv,
closing the loop: the phone screens locally, reports spam (and user-consented corrections), and
this brings both back into the training set.

Two kinds of row come back now:
  * label 1 - spam the on-device model flagged in the wild.
  * label 0 - FALSE POSITIVES the user corrected AND consented to share. These are the valuable
              ones: they are precisely the mistakes the current model makes, which no amount of
              generated data can predict.

The backend only exports rows that clear MIN_DEVICE_COUNT independent reporters, so a single
malicious install cannot inject training data. Do not bypass that by lowering the threshold in
production - the app's API key ships inside the APK and can be extracted.

Config (env vars, or ml_pipeline/.env):
    TELEMETRY_SERVER_URL
    TELEMETRY_API_KEY
"""
import argparse
import os
import sys

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RAW_DIR = "data/raw"
EXPORT_PATH = os.path.join(RAW_DIR, "telemetry_export.csv")
CURSOR_PATH = os.path.join(RAW_DIR, ".last_telemetry_pull")


def get_cursor() -> int:
    if os.path.exists(CURSOR_PATH):
        with open(CURSOR_PATH, "r") as f:
            return int(f.read().strip() or 0)
    return 0


def set_cursor(timestamp_ms: int):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CURSOR_PATH, "w") as f:
        f.write(str(timestamp_ms))


def fetch(server_url: str, api_key: str, since_ms: int, min_devices=None) -> dict:
    params = {"since": since_ms, "format": "json"}
    if min_devices is not None:
        params["minDeviceCount"] = min_devices
    resp = requests.get(
        f"{server_url.rstrip('/')}/api/telemetry/export",
        params=params, headers={"x-api-key": api_key}, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Re-pull everything, ignoring the saved cursor.")
    parser.add_argument("--min-devices", type=int, default=None,
                        help="Override the server's independent-reporter threshold (testing only).")
    args = parser.parse_args()

    server_url = os.environ.get("TELEMETRY_SERVER_URL")
    api_key = os.environ.get("TELEMETRY_API_KEY")
    if not server_url or not api_key:
        print("Set TELEMETRY_SERVER_URL and TELEMETRY_API_KEY first.")
        sys.exit(1)

    since_ms = 0 if args.full else get_cursor()
    print(f"Fetching telemetry since {since_ms} ...")
    payload = fetch(server_url, api_key, since_ms, args.min_devices)
    samples = payload.get("samples", [])

    print(f"  spam reported      : {payload.get('spamCount', 0)}")
    print(f"  ham corrections    : {payload.get('hamCorrectionCount', 0)}")
    print(f"  min device count   : {payload.get('minDeviceCount')}")

    if not samples:
        print("Nothing new since the last pull.")
        return

    new_df = pd.DataFrame([{"text": s["text"], "label": int(s["label"])} for s in samples
                           if s.get("text")])
    if new_df.empty:
        print("No usable text in the response.")
        return

    if os.path.exists(EXPORT_PATH):
        existing = pd.read_csv(EXPORT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # A message can legitimately flip label after a user correction; keep the newest verdict.
    combined = combined.drop_duplicates(subset=["text"], keep="last")

    os.makedirs(RAW_DIR, exist_ok=True)
    combined.to_csv(EXPORT_PATH, index=False, encoding="utf-8")

    latest_ms = max(int(pd.Timestamp(s["lastSeen"]).timestamp() * 1000) for s in samples)
    set_cursor(latest_ms)

    spam_n = int((combined.label == 1).sum())
    ham_n = int((combined.label == 0).sum())
    print(f"Merged {len(new_df)} new row(s). telemetry_export.csv now holds "
          f"{len(combined)} unique ({spam_n} spam / {ham_n} corrected-ham).")
    print("Next: python preprocess.py && python train.py && python quality_gate.py")


if __name__ == "__main__":
    main()

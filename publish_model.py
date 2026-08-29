"""
Registers an already-gated, already-uploaded model with the backend so phones start picking it up.

Order matters and is deliberate: the CI workflow uploads the .tflite to a durable URL (a GitHub
Release asset) FIRST, then calls this. If the manifest were published before the file existed,
every device would poll for a model that 404s.

This never uploads weights to the backend. It stores a pointer plus the sha256 the app verifies
after downloading, so a corrupted or tampered download is rejected on-device rather than silently
classifying with garbage.

Requires PUBLISH_KEY - deliberately a different secret from the app's API key, which ships inside
the APK and can be extracted from any installed copy.
"""
import argparse
import hashlib
import json
import os
import sys

import requests

MODELS_DIR = "models"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-url", required=True,
                    help="Public https URL the app will download from (GitHub Release asset).")
    ap.add_argument("--model-path", default=os.path.join(MODELS_DIR, "spam_detector.tflite"),
                    help="Local copy of the exact bytes uploaded, for checksumming.")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--trained-up-to", default=None,
                    help="ISO timestamp of the newest telemetry sample used; the backend uses "
                         "this as the cursor for the next retrain trigger.")
    ap.add_argument("--min-app-version-code", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    server_url = os.environ.get("TELEMETRY_SERVER_URL")
    publish_key = os.environ.get("PUBLISH_KEY")
    if not server_url or not publish_key:
        print("Set TELEMETRY_SERVER_URL and PUBLISH_KEY.")
        sys.exit(1)

    gate_path = os.path.join(MODELS_DIR, "gate_report.json")
    if not os.path.exists(gate_path):
        print("gate_report.json missing - run quality_gate.py before publishing.")
        sys.exit(1)
    gate = json.load(open(gate_path, encoding="utf-8"))
    if not gate.get("passed"):
        print("Refusing to publish: the quality gate did not pass.")
        for f in gate.get("failures", []):
            print(f"  * {f}")
        sys.exit(1)

    meta = json.load(open(os.path.join(MODELS_DIR, "model_meta.json"), encoding="utf-8"))
    digest = sha256_file(args.model_path)

    payload = {
        "modelUrl": args.model_url,
        "sha256": digest,
        "threshold": float(meta["threshold"]),
        "featureContract": {
            "numBuckets": int(meta["num_buckets"]),
            "maxFeatures": int(meta["max_features"]),
        },
        "metrics": {
            "realF1": gate["realF1"],
            "realPrecision": gate["realPrecision"],
            "realRecall": gate["realRecall"],
        },
        "notes": args.notes,
        "minAppVersionCode": args.min_app_version_code,
    }
    if args.trained_up_to:
        payload["trainedUpTo"] = args.trained_up_to

    print(f"sha256 {digest}")
    print(f"url    {args.model_url}")
    print(f"real F1 {gate['realF1']:.4f}  P {gate['realPrecision']:.4f}  R {gate['realRecall']:.4f}")

    if args.dry_run:
        print("\n--dry-run: not contacting the backend. Payload would be:")
        print(json.dumps(payload, indent=2))
        return

    resp = requests.post(f"{server_url.rstrip('/')}/api/model/publish",
                         json=payload, headers={"x-publish-key": publish_key}, timeout=30)
    if resp.status_code != 201:
        print(f"Publish failed: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)

    version = resp.json().get("version")
    print(f"\nPublished as model v{version}. Devices will pick it up on their next check.")

    # Local record of what is live, so quality_gate.py still has a regression baseline even when
    # the backend is unreachable on a later run.
    with open(os.path.join(MODELS_DIR, "published_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**payload["metrics"], "version": version, "sha256": digest}, f, indent=2)


if __name__ == "__main__":
    main()

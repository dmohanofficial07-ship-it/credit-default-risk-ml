"""Download and validate the research dataset directly from UCI."""

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FEATURES = ["LIMIT_BAL", *STATUS,
            *[f"BILL_AMT{i}" for i in range(1, 7)],
            *[f"PAY_AMT{i}" for i in range(1, 7)]]
ORIGINAL_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                  *STATUS, *[f"BILL_AMT{i}" for i in range(1, 7)],
                  *[f"PAY_AMT{i}" for i in range(1, 7)]]
RENAME = {f"X{i}": name for i, name in enumerate(ORIGINAL_NAMES, 1)}


def parse_data(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    expected = {"ID", "Y", *RENAME}
    if set(reader.fieldnames or []) != expected:
        raise ValueError("UCI schema changed; inspect the data before training.")
    rows = list(reader)
    if len(rows) != 30000:
        raise ValueError(f"Expected 30000 source rows, received {len(rows)}.")
    ids = np.array([int(row["ID"]) for row in rows])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate source IDs.")
    source_keys = {name: key for key, name in RENAME.items()}
    X = np.array([[float(row[source_keys[name]]) for name in FEATURES] for row in rows])
    y = np.array([float(row["Y"]) for row in rows])
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected missing/nonfinite feature or invalid target.")
    if not np.equal(X[:, 1:7], np.floor(X[:, 1:7])).all():
        raise ValueError("Repayment status codes must be integers.")
    return X, y.astype(int), ids


def load_data():
    csv_path, provenance_path = DATA / "uci_credit_default.csv", DATA / "source.json"
    if not csv_path.exists():
        print("Downloading the dataset directly from UCI...", flush=True)
        with urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
        parse_data(raw)  # Validate before accepting a downloaded file.
        DATA.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(raw)
        provenance = {
            "dataset": "Default of Credit Card Clients",
            "creator": "I-Cheng Yeh",
            "doi": "10.24432/C55S3H",
            "source_page": SOURCE_URL,
            "download_url": DATA_URL,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "history_period": "April-September 2005",
            "population": "Credit card clients in Taiwan",
        }
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    raw = csv_path.read_bytes()
    if not provenance_path.exists():
        raise ValueError("Missing source.json: restore provenance before training.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != provenance["sha256"]:
        raise ValueError("Dataset differs from the recorded download checksum.")
    return (*parse_data(raw), provenance)


if __name__ == "__main__":
    X, y, ids, provenance = load_data()
    print(f"Validated {len(y):,} records, {X.shape[1]} selected features.")
    print(f"Observed default rate: {y.mean():.2%}")
    print(f"Source: {SOURCE_URL}")
    print(f"SHA-256: {provenance['sha256']}")

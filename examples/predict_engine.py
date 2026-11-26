"""Send a real FD001 engine history without optional client dependencies."""

import argparse
import json
import urllib.error
import urllib.request

import pandas as pd

from predictive_maintenance.config import RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    args = parser.parse_args()
    data = pd.read_csv(RAW_DATA_DIR / "test_data.csv")
    engine = data.loc[data.unit == args.unit].sort_values("cycle")
    if engine.empty:
        parser.error(f"Engine {args.unit} not found")
    payload = {
        "unit": f"engine_{args.unit:03d}",
        "readings": engine.drop(columns=["unit", "cycle"]).to_dict("records"),
    }
    request = urllib.request.Request(
        args.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(json.dumps(json.load(response), indent=2))
    except urllib.error.HTTPError as exc:
        parser.exit(1, f"API returned {exc.code}: {exc.read().decode()}\n")


if __name__ == "__main__":
    main()

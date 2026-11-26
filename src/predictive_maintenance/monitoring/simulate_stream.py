from datetime import datetime, timezone
import json
import time

from loguru import logger
import pandas as pd
import requests
import typer

from predictive_maintenance.config import RAW_DATA_DIR, RESULTS_DIR

app = typer.Typer()

LOG_PATH = RESULTS_DIR / "monitoring" / "logs" / "predictions.jsonl"


def build_windows(test_raw: pd.DataFrame) -> list[tuple[int, list[dict]]]:
    raw_cols = [c for c in test_raw.columns if c not in ("unit", "cycle")]
    windows = []
    for unit, unit_df in test_raw.sort_values("cycle").groupby("unit"):
        readings = unit_df[raw_cols].to_dict(orient="records")
        windows.append((int(unit), readings))
    return windows


def inject_drift(readings: list[dict], severity: float = 3.0) -> list[dict]:
    return [{k: v * severity for k, v in r.items()} for r in readings]


@app.command()
def main(
    api_url: str = "http://localhost:8000/predict",
    rate: float = 2.0,
    inject_drift_flag: bool = typer.Option(False, "--inject-drift"),
    drift_after: int = 50,
    drift_severity: float = 3.0,
    loop: bool = False,
) -> None:
    if rate <= 0:
        raise typer.BadParameter("rate must be positive")
    test_raw = pd.read_csv(RAW_DATA_DIR / "test_data.csv")
    windows = build_windows(test_raw)
    logger.info(f"{len(windows)} moteurs de test chargés, envoi à {rate} req/s vers {api_url}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_sent = 0
    while True:
        for unit_id, readings in windows:
            drift_active = inject_drift_flag and n_sent >= drift_after
            payload_readings = inject_drift(readings, drift_severity) if drift_active else readings

            payload = {"unit": f"engine_{unit_id:03d}", "readings": payload_readings}

            try:
                response = requests.post(api_url, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                status = "ok"
            except requests.RequestException as exc:
                result = {}
                status = f"error: {exc}"

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unit_id": payload["unit"],
                "status": status,
                "drift_injected": drift_active,
                "last_reading": payload_readings[-1] if payload_readings else None,
                **result,
            }

            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            n_sent += 1
            flag = " [DRIFT INJECTÉ]" if drift_active else ""
            logger.info(
                f"[{n_sent}] {payload['unit']} -> {status}{flag} "
                f"(RUL={result.get('predicted_rul', 'NA')}, classe={result.get('predicted_class', 'NA')})"
            )

            time.sleep(1.0 / rate)

        if not loop:
            break

    logger.success(f"Terminé — {n_sent} requêtes envoyées, logs dans {LOG_PATH}")


if __name__ == "__main__":
    app()

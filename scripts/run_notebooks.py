"""Execute the maintained notebooks in order, with a fresh project kernel for each."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda.ipynb",
    ROOT / "notebooks" / "02_data_cleaning.ipynb",
    ROOT / "notebooks" / "03_feature_engineering.ipynb",
    ROOT / "notebooks" / "04_model_training.ipynb",
    ROOT / "notebooks" / "05_model_evaluation.ipynb",
    ROOT / "notebooks" / "EDA_resultats_maintenance_predictive.ipynb",
    ROOT / "Training.ipynb",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Optional notebook paths")
    parser.add_argument(
        "--in-place", action="store_true", help="Save executed outputs to source notebooks"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".uv-cache" / "notebook-runs")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds per code cell")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    try:
        from nbclient import NotebookClient
        import nbformat
    except ImportError:
        parser.exit(1, "Install notebook dependencies: uv sync --locked --extra notebooks\n")

    paths = [path.resolve() for path in args.notebooks] or NOTEBOOKS
    for path in paths:
        if not path.is_file():
            parser.error(f"Notebook is missing: {path}")
    for path in paths:
        relative = path.relative_to(ROOT)
        print(f"Running {relative} ...", flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        # Avoid a globally registered kernel accidentally using another environment.
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        try:
            client.execute()
        except Exception:
            failed = args.output_dir / relative.with_suffix(".failed.ipynb")
            failed.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, failed)
            print(f"Failed execution saved to {failed}", file=sys.stderr)
            raise
        output = path if args.in_place else args.output_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        print(f"Passed {relative}", flush=True)
    print(f"All {len(paths)} notebooks executed successfully.")


if __name__ == "__main__":
    main()

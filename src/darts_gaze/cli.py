"""Command-line entrypoints for the darts gaze pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_DB_PATH, PROCESSED_DIR
from .dataset import DatasetBuilder
from .modeling import train_baselines
from .sportradar import SportradarClient
from .storage import AnnotationStore
from .webapp import create_app


def annotate_main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run the darts gaze annotation web app.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the Flask app to.")
    parser.add_argument("--port", default=5000, type=int, help="Port to bind the Flask app to.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args(argv)

    app = create_app(db_path=args.db)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build processed datasets from stored frame captures.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory for processed outputs.")
    parser.add_argument("--no-write-back", action="store_true", help="Do not update capture rows with auto-match results.")
    args = parser.parse_args(argv)

    store = AnnotationStore(args.db)
    client = SportradarClient()
    builder = DatasetBuilder(store=store, client=client, output_dir=args.output_dir)
    artifacts = builder.build(write_back=not args.no_write_back)
    print(json.dumps(asdict(artifacts), indent=2, default=str))
    return 0


def train_baselines_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline darts gaze models.")
    parser.add_argument(
        "--dataset",
        default=str(Path(PROCESSED_DIR) / "training_samples.csv"),
        help="Path to the processed training dataset CSV.",
    )
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory for model artifacts and plots.")
    args = parser.parse_args(argv)

    metrics = train_baselines(dataset_csv=args.dataset, output_dir=args.output_dir)
    print(json.dumps(metrics, indent=2))
    return 0

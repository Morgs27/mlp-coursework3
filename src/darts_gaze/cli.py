"""Command-line entrypoints for the darts gaze pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_DB_PATH, FIRST_PASS_OUTPUTS_DIR, PROCESSED_DIR
from .dataset import DatasetBuilder
from .modeling import train_baselines
from .paper_figures import export_paper_figures
from .reporting import export_quality_reports
from .sportradar import SportradarClient
from .storage import AnnotationStore
from .webapp import create_app


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _build_dataset(db: str, output_dir: str, write_back: bool) -> dict:
    store = AnnotationStore(db)
    client = SportradarClient()
    builder = DatasetBuilder(store=store, client=client, output_dir=output_dir)
    artifacts = builder.build(write_back=write_back)
    return asdict(artifacts)


def _export_qa(processed_dir: str, report_dir: str) -> dict:
    artifacts = export_quality_reports(processed_dir=processed_dir, report_dir=report_dir)
    return asdict(artifacts)


def _evaluate(dataset_csv: str, model_dir: str, report_dir: str) -> dict:
    return train_baselines(dataset_csv=dataset_csv, output_dir=model_dir, report_dir=report_dir)


def _paper_figures(processed_dir: str, report_dir: str, output_dir: str) -> dict:
    artifacts = export_paper_figures(processed_dir=processed_dir, report_dir=report_dir, output_dir=output_dir)
    return asdict(artifacts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the darts gaze collection and modeling pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate_parser = subparsers.add_parser("annotate", help="Run the darts gaze annotation web app.")
    annotate_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    annotate_parser.add_argument("--host", default="127.0.0.1", help="Host to bind the Flask app to.")
    annotate_parser.add_argument("--port", default=5000, type=int, help="Port to bind the Flask app to.")
    annotate_parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")

    build_parser_cmd = subparsers.add_parser("build-dataset", help="Build processed datasets from stored frame captures.")
    build_parser_cmd.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    build_parser_cmd.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory for processed outputs.")
    build_parser_cmd.add_argument("--no-write-back", action="store_true", help="Do not update capture rows with auto-match results.")

    qa_parser = subparsers.add_parser("qa", help="Build the dataset and export first-pass QA tables and figures.")
    qa_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    qa_parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory for processed dataset outputs.")
    qa_parser.add_argument("--report-dir", default=str(FIRST_PASS_OUTPUTS_DIR), help="Directory for first-pass tables and figures.")
    qa_parser.add_argument("--no-write-back", action="store_true", help="Do not update capture rows with auto-match results.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run first-pass baseline training and evaluation.")
    evaluate_parser.add_argument(
        "--dataset",
        default=str(Path(PROCESSED_DIR) / "training_samples.csv"),
        help="Path to the processed training dataset CSV.",
    )
    evaluate_parser.add_argument("--model-dir", default=str(PROCESSED_DIR), help="Directory for fitted model artifacts.")
    evaluate_parser.add_argument("--report-dir", default=str(FIRST_PASS_OUTPUTS_DIR), help="Directory for first-pass tables and figures.")

    paper_parser = subparsers.add_parser("paper-figures", help="Generate paper-ready figures from the latest processed data and evaluation tables.")
    paper_parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory containing processed dataset CSVs.")
    paper_parser.add_argument("--report-dir", default=str(FIRST_PASS_OUTPUTS_DIR), help="Directory containing first-pass evaluation tables.")
    paper_parser.add_argument("--output-dir", default="outputs/paper", help="Directory for paper-ready tables and figures.")

    first_pass_parser = subparsers.add_parser("first-pass", help="Run dataset build, QA export, and evaluation end to end.")
    first_pass_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the annotation SQLite database.")
    first_pass_parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Directory for processed dataset outputs.")
    first_pass_parser.add_argument("--report-dir", default=str(FIRST_PASS_OUTPUTS_DIR), help="Directory for first-pass tables and figures.")
    first_pass_parser.add_argument("--no-write-back", action="store_true", help="Do not update capture rows with auto-match results.")

    return parser


def annotate_main(argv: list[str] | None = None) -> int:
    _maybe_load_dotenv()
    parser = build_parser()
    args = parser.parse_args(["annotate", *(argv or [])])
    app = create_app(db_path=args.db)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_dataset_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(["build-dataset", *(argv or [])])
    print(json.dumps(_build_dataset(args.db, args.output_dir, write_back=not args.no_write_back), indent=2, default=str))
    return 0


def train_baselines_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(["evaluate", *(argv or [])])
    metrics = _evaluate(args.dataset, args.model_dir, args.report_dir)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    _maybe_load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotate":
        app = create_app(db_path=args.db)
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    if args.command == "build-dataset":
        print(json.dumps(_build_dataset(args.db, args.output_dir, write_back=not args.no_write_back), indent=2, default=str))
        return 0

    if args.command == "qa":
        build_artifacts = _build_dataset(args.db, args.processed_dir, write_back=not args.no_write_back)
        qa_artifacts = _export_qa(args.processed_dir, args.report_dir)
        print(json.dumps({"build": build_artifacts, "qa": qa_artifacts}, indent=2, default=str))
        return 0

    if args.command == "evaluate":
        metrics = _evaluate(args.dataset, args.model_dir, args.report_dir)
        print(json.dumps(metrics, indent=2, default=str))
        return 0

    if args.command == "paper-figures":
        artifacts = _paper_figures(args.processed_dir, args.report_dir, args.output_dir)
        print(json.dumps(artifacts, indent=2, default=str))
        return 0

    if args.command == "first-pass":
        build_artifacts = _build_dataset(args.db, args.processed_dir, write_back=not args.no_write_back)
        qa_artifacts = _export_qa(args.processed_dir, args.report_dir)
        metrics = _evaluate(
            dataset_csv=str(Path(args.processed_dir) / "training_samples.csv"),
            model_dir=args.processed_dir,
            report_dir=args.report_dir,
        )
        print(json.dumps({"build": build_artifacts, "qa": qa_artifacts, "evaluation": metrics}, indent=2, default=str))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

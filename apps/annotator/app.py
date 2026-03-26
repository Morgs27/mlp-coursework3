"""Thin wrapper for running the annotator from the repository app folder."""

from darts_gaze.cli import annotate_main


if __name__ == "__main__":
    raise SystemExit(annotate_main())

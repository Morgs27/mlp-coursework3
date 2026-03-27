set shell := ["bash", "-cu"]

db := "data/annotations.sqlite3"
processed_dir := "data/processed"
report_dir := "outputs/first_pass"
paper_dir := "outputs/paper"

annotate:
    PYTHONPATH=src python -m darts_gaze.cli annotate --db {{db}}

build-dataset:
    PYTHONPATH=src python -m darts_gaze.cli build-dataset --db {{db}} --output-dir {{processed_dir}}

qa:
    PYTHONPATH=src python -m darts_gaze.cli qa --db {{db}} --processed-dir {{processed_dir}} --report-dir {{report_dir}}

evaluate:
    PYTHONPATH=src python -m darts_gaze.cli evaluate --dataset {{processed_dir}}/training_samples.csv --model-dir {{processed_dir}} --report-dir {{report_dir}}

paper-figures:
    PYTHONPATH=src python -m darts_gaze.cli paper-figures --processed-dir {{processed_dir}} --report-dir {{report_dir}} --output-dir {{paper_dir}}

first-pass:
    PYTHONPATH=src python -m darts_gaze.cli first-pass --db {{db}} --processed-dir {{processed_dir}} --report-dir {{report_dir}}

paper:
    just first-pass
    just paper-figures

replot:
    just evaluate
    just paper-figures

test:
    PYTHONPATH=src pytest -q

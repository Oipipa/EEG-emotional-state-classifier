.PHONY: install lint test format serve pipeline

install:
\tpip install -e .

lint:
\truff src tests

test:
\tpytest

serve:
\tuvicorn eeg_emotion.api:create_app --factory --reload

pipeline:
\teeg-emotion pipeline

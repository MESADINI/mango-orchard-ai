# Thin wrapper around tasks.py (which also works on Windows without make).
# Override the interpreter with:  make PYTHON=backend/venv/bin/python <target>
PYTHON ?= python

.PHONY: env data train-disease train-pest eval test all

env data train-disease train-pest eval test all:
	$(PYTHON) tasks.py $@

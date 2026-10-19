PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY) -m

.PHONY: all fetch clean growth subareas validate test refetch

all: fetch clean growth subareas

fetch:
	$(RUN) clbib.fetch

refetch:
	$(RUN) clbib.fetch --force

clean:
	$(RUN) clbib.clean

growth:
	$(RUN) clbib.growth

subareas:
	$(RUN) clbib.subareas

validate:
	$(RUN) clbib.validate score

test:
	PYTHONPATH=src $(PY) -m unittest discover -s tests -v

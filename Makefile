PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY) -m

.PHONY: all fetch clean growth test refetch

all: fetch clean growth

fetch:
	$(RUN) clbib.fetch

refetch:
	$(RUN) clbib.fetch --force

clean:
	$(RUN) clbib.clean

growth:
	$(RUN) clbib.growth

test:
	PYTHONPATH=src $(PY) -m unittest discover -s tests -v

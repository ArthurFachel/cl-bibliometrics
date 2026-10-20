PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY) -m

.PHONY: all fetch clean growth subareas validate test refetch

all: fetch clean growth subareas validate

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

# Rule scores under each labeller, plus pairwise agreement, from the label
# files already in validation/. New labels need an API key (validation/README.md).
validate:
	$(RUN) clbib.validate score labels_llm.csv llm_dev
	$(RUN) clbib.validate score labels_llm_test.csv llm_test
	$(RUN) clbib.validate score labels_deepseek.csv deepseek_dev
	$(RUN) clbib.validate score labels_deepseek_test.csv deepseek_test
	$(RUN) clbib.validate score labels_sonnet.csv sonnet_dev
	$(RUN) clbib.validate score labels_sonnet_test.csv sonnet_test
	$(RUN) clbib.llm_label compare --split dev
	$(RUN) clbib.llm_label compare --split test

test:
	PYTHONPATH=src $(PY) -m unittest discover -s tests -v

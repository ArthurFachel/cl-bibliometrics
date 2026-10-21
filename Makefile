PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY) -m

.PHONY: all fetch clean growth recall subareas validate networks benchmarks drivers sensitivity report test refetch

all: fetch clean growth recall subareas validate networks benchmarks drivers sensitivity

fetch:
	$(RUN) clbib.fetch

refetch:
	$(RUN) clbib.fetch --force

clean:
	$(RUN) clbib.clean

growth:
	$(RUN) clbib.growth

recall:
	$(RUN) clbib.recall score

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
	$(RUN) clbib.country_validate score
	$(RUN) clbib.benchmark_validate score
	$(RUN) clbib.nlp_validate score

test:
	PYTHONPATH=src $(PY) -m unittest discover -s tests -v

networks:
	$(RUN) clbib.networks

benchmarks:
	$(RUN) clbib.benchmarks

drivers:
	$(RUN) clbib.drivers

# docs/report.pdf from docs/report.md (needs a python3 with `markdown` and LibreOffice; not part of all)
report:
	scripts/build_report.sh

# Same analyses with repository-only works kept (results/sensitivity_with_repository_only/),
# then results/sensitivity.csv compares the headline numbers.
sensitivity:
	CLBIB_INCLUDE_REPOSITORY_ONLY=1 $(RUN) clbib.growth
	CLBIB_INCLUDE_REPOSITORY_ONLY=1 $(RUN) clbib.subareas
	CLBIB_INCLUDE_REPOSITORY_ONLY=1 $(RUN) clbib.networks
	CLBIB_INCLUDE_REPOSITORY_ONLY=1 $(RUN) clbib.benchmarks
	$(RUN) clbib.subareas
	$(RUN) clbib.sensitivity

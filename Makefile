# Regenerate every table, figure copy and macro the paper uses, then build the PDF.
#   make tables   all tables from released runs and results/revision/
#   make pdf      paper/main.pdf (tectonic)
#   make diff     paper/diff.pdf against the submitted version (needs latexdiff)

PY      := $(CURDIR)/.venv/bin/python
BENCH   := SCFP/benchmark
TABLES  := $(CURDIR)/paper/paper_tables
RESULTS := $(CURDIR)/results/revision

.PHONY: tables pdf diff

tables:
	cd $(BENCH) && PYTHONPATH=. $(PY) -m scfp_bench.experiments.make_tables --config configs/laptop_local.yaml --out $(TABLES)
	cd $(BENCH) && PYTHONPATH=. $(PY) -m scfp_bench.experiments.make_tables_v7 --out $(TABLES)
	cd $(BENCH) && PYTHONPATH=. $(PY) -m scfp_bench.experiments.make_tables_revision --results $(RESULTS) --out $(TABLES)

pdf:
	cd paper && tectonic -X compile main.tex

diff:
	cd paper && latexdiff --flatten submitted_v7.tex main.tex > diff.tex && tectonic -X compile diff.tex

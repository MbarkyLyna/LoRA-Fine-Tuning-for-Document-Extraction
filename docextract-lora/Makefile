.PHONY: real install data bench report test all lint clean
install:
	pip install -e ".[dev]"
data:
	python -m docextract.synth --out data
bench: data
	python -m docextract.bench
real:
	bash scripts/prepare_sroie.sh
	python -m docextract.real_sroie --evaluate
report:
	python -m docextract.report
test:
	pytest -q
lint:
	ruff check .
all: test bench report
clean:
	rm -rf data runs .pytest_cache

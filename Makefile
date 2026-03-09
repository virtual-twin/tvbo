# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

IMAGE_NAME=tvbo
IMAGE_TAG=latest
IMAGE_FULL=$(IMAGE_NAME):$(IMAGE_TAG)
TARBALL_PATH=/Users/leonmartin_bih/projects/TVB-O/tvbo-container/tvbo.tar.gz

.PHONY: build save run docs-quarto docs-jupyter docs-to-py docs-rm-py docs-test docs-pytest docs-pytest-all docs-test-all docs-preview docs-render docs-publish pypi-release release gen-linkml gen-openminds all
all: build save

# LinkML schema generation
SCHEMA_PATH = schema/tvbo_datamodel.yaml
DATAMODEL_DIR = tvbo/datamodel
OPENMINDS_DIR = schema/openMINDS_tvbo

gen-linkml:
	@echo "Generating Python datamodel from LinkML schema..."
	@mkdir -p $(DATAMODEL_DIR)
	@gen-pydantic $(SCHEMA_PATH) > $(DATAMODEL_DIR)/tvbopydantic.py
	@gen-python $(SCHEMA_PATH) > $(DATAMODEL_DIR)/tvbo_datamodel.py
	@echo "✓ LinkML datamodel generated in $(DATAMODEL_DIR)/"

gen-openminds:
	@echo "Generating openMINDS schemas from LinkML source..."
	@python $(OPENMINDS_DIR)/generate_openminds.py
	@echo "✓ openMINDS schemas generated in $(OPENMINDS_DIR)/schemas/"

gen-all: gen-linkml gen-openminds
	@echo "✓ All schemas generated"

build:
	DOCKER_BUILDKIT=1 docker build --secret id=gitlab_token,env=GITLAB_TOKEN -t $(IMAGE_FULL) .

save:
	docker save $(IMAGE_FULL) | gzip > $(TARBALL_PATH)

run:
	docker run -it --rm -e MODE=jupyter -p 8888:8888 $(IMAGE_FULL)

docs-quarto:
	find ./docs/Usage -name '*.ipynb' -exec quarto convert {} \; && find ./docs/Usage -name '*.ipynb' -exec rm {} \;

docs-jupyter:
	find ./docs/Usage -name '*.qmd' -exec quarto convert {} \; && find ./docs/Usage -name '*.qmd' -exec rm {} \;

docs-to-py:
	find ./docs/Usage -name '*.ipynb' -exec jupytext --to py:percent {} \;

docs-rm-py:
	find ./docs/Usage -name '*.py' -exec rm {} \;

DOCS_TEST_JOBS ?= 4

docs-test:
	@echo "Testing all .qmd files in docs/ ($(DOCS_TEST_JOBS) parallel jobs)..."
	@echo "========================================"
	@resultsdir=$$(mktemp -d); \
	test_one() { \
		qmd="$$1"; resdir="$$2"; \
		name=$$(basename "$$qmd"); \
		if ! grep -q '```{python}' "$$qmd"; then \
			echo "⊘ SKIPPED (no python cells): $$qmd"; \
			echo "skip" > "$$resdir/$${name}.result"; \
			return; \
		fi; \
		ipynb="$${qmd%.qmd}.ipynb"; \
		if ! quarto convert "$$qmd" --output "$$ipynb" 2>/dev/null; then \
			echo "✗ FAILED (quarto convert): $$qmd"; \
			echo "fail:quarto convert failed" > "$$resdir/$${name}.result"; \
			return; \
		fi; \
		errlog=$$(mktemp); \
		if MPLBACKEND=Agg jupyter execute "$$ipynb" --allow-errors 2>"$$errlog"; then \
			echo "✓ PASSED: $$qmd"; \
			echo "pass" > "$$resdir/$${name}.result"; \
		else \
			errmsg=$$(tail -1 "$$errlog" | head -c 100); \
			echo "✗ FAILED: $$qmd - $$errmsg"; \
			echo "fail:$$errmsg" > "$$resdir/$${name}.result"; \
		fi; \
		rm -f "$$ipynb" "$$errlog"; \
	}; \
	export -f test_one; \
	find ./docs -name '*.qmd' -type f | sort | \
		xargs -P $(DOCS_TEST_JOBS) -I {} bash -c 'test_one "$$1" "$$2"' _ {} "$$resultsdir"; \
	echo ""; \
	echo "========================================"; \
	passed=$$(grep -l '^pass$$' "$$resultsdir"/*.result 2>/dev/null | wc -l | tr -d ' '); \
	skipped=$$(grep -l '^skip$$' "$$resultsdir"/*.result 2>/dev/null | wc -l | tr -d ' '); \
	failed=$$(grep -l '^fail:' "$$resultsdir"/*.result 2>/dev/null | wc -l | tr -d ' '); \
	total=$$((passed + skipped + failed)); \
	echo "Test Summary: $$passed passed, $$failed failed, $$skipped skipped ($$total total)"; \
	if [ "$$failed" -gt 0 ]; then \
		echo ""; \
		echo "Failed tests:"; \
		for f in "$$resultsdir"/*.result; do \
			if grep -q '^fail:' "$$f"; then \
				name=$$(basename "$$f" .result); \
				reason=$$(cat "$$f" | sed 's/^fail://'); \
				echo "  $$name: $$reason"; \
			fi; \
		done; \
	fi; \
	echo "========================================"; \
	rm -rf "$$resultsdir"; \
	[ "$$failed" -eq 0 ]

# Pytest-based docs testing (for CI/CD)
# Requires: pip install pytest-xdist
docs-pytest:
	@echo "Running documentation tests with pytest ($(DOCS_TEST_JOBS) workers)..."
	pytest tests/test_docs.py -v -x --tb=short -n $(DOCS_TEST_JOBS)

docs-pytest-all:
	@echo "Running all documentation tests ($(DOCS_TEST_JOBS) workers, no early exit)..."
	pytest tests/test_docs.py -v --tb=short -n $(DOCS_TEST_JOBS)

docs-test-all: docs-jupyter docs-test docs-quarto
	@echo "Full test pipeline completed!"

docs-gen-datamodel:
	@echo "Generating LinkML datamodel documentation..."
	@cd docs && python scripts/generate_datamodel_docs.py
	@echo "✓ DataModel documentation generated in docs/datamodel/"

docs-preview:
	@echo "Starting Quarto preview (freeze: auto caches notebooks)..."
	@cd docs && quarto preview

docs-render:
	@echo "Full Quarto render..."
	@cd docs && quarto render

docs-publish:
	@echo "Publishing docs to GitHub Pages..."
	@cd docs && quarto publish gh-pages --no-prompt

docs-test-to-debug:
	@mkdir -p ./docs/Usage
	@echo "Testing debugged files in docs/to_debug..."
	@echo "========================================"
	@passed=0; failed=0; \
	find ./docs/to_debug -name '*.ipynb' -type f | while read notebook; do \
		echo ""; \
		echo "Testing: $$notebook"; \
		if MPLBACKEND=Agg jupyter nbconvert --execute --to notebook --inplace "$$notebook" > /dev/null 2>&1; then \
			echo "✓ PASSED - Moving back to docs/Usage"; \
			relpath=$$(echo "$$notebook" | sed 's|./docs/to_debug/||'); \
			targetdir=$$(dirname "./docs/Usage/$$relpath"); \
			mkdir -p "$$targetdir"; \
			mv "$$notebook" "./docs/Usage/$$relpath"; \
			qmdfile="$${notebook%.ipynb}.qmd"; \
			if [ -f "$$qmdfile" ]; then \
				mv "$$qmdfile" "$${targetdir}/$$(basename $$qmdfile)"; \
			fi; \
			passed=$$((passed + 1)); \
		else \
			echo "✗ STILL FAILING - Keeping in to_debug"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "Debug Test Summary:"; \
	echo "  Fixed & Moved: $$passed"; \
	echo "  Still Failing: $$failed"; \
	echo "========================================"

pypi-release:
	@echo "Building and uploading to PyPI..."
	@rm -rf dist build *.egg-info
	@python -m build
	@python -m twine check dist/*
	@python -m twine upload dist/*
	@echo "✓ Release uploaded to PyPI"

release:
	@echo "Creating GitHub release..."
	@VERSION=$$(grep '^__version__' tvbo/__init__.py | cut -d'"' -f2); \
	echo "Current version: $$VERSION"; \
	git add -A; \
	git commit -m "Release v$$VERSION" || true; \
	git push; \
	gh release create "v$$VERSION" \
		--title "v$$VERSION" \
		--notes "See CHANGELOG.md for details" \
		--generate-notes; \
	echo "✓ GitHub release v$$VERSION created"
	@echo "✓ GitHub Actions will automatically publish to PyPI"


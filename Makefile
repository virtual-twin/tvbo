# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

IMAGE_NAME=tvbo
IMAGE_TAG=latest
IMAGE_FULL=$(IMAGE_NAME):$(IMAGE_TAG)
TARBALL_PATH=/Users/leonmartin_bih/projects/TVB-O/tvbo-container/tvbo.tar.gz

.PHONY: help build save run docs-quarto docs-jupyter docs-to-py docs-rm-py docs-test docs-pytest docs-pytest-all docs-test-all docs-preview docs-render docs-clean docs-publish docs-publish-changed pypi-release release gen-linkml gen-openminds gen-owl gen-shacl gen-all all

help: ## Show this help
	@echo "TVBO Makefile"
	@echo "============="
	@echo ""
	@echo "Docker:"
	@echo "  make build              Build Docker image"
	@echo "  make save               Save Docker image to tarball"
	@echo "  make run                Run Docker container (Jupyter mode)"
	@echo ""
	@echo "Schema Generation:"
	@echo "  make gen-linkml         Generate Python datamodel from LinkML schema"
	@echo "  make gen-openminds      Generate openMINDS schemas from LinkML"
	@echo "  make gen-owl            Generate OWL ontology (tvb-o-struct.owl) from LinkML"
	@echo "  make gen-shacl          Generate SHACL shapes (tvb-o.shacl.ttl) from LinkML"
	@echo "  make gen-all            Run all schema generators"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-preview       Preview docs (pre-render once, then live reload)"
	@echo "  make docs-render        Full Quarto render"
	@echo "  make docs-clean         Remove generated docs (api/, datamodel/, _site/)"
	@echo "  make docs-publish       Publish docs to GitHub Pages (full render)"
	@echo "  make docs-publish-changed Render only changed .qmd files, then publish"
	@echo "  make docs-gen-datamodel Generate LinkML datamodel documentation"
	@echo "  make docs-quarto        Convert Usage notebooks (.ipynb) to .qmd"
	@echo "  make docs-jupyter       Convert Usage .qmd files to .ipynb"
	@echo "  make docs-to-py         Convert Usage notebooks to .py (percent format)"
	@echo "  make docs-rm-py         Remove .py files from Usage"
	@echo ""
	@echo "Documentation Testing:"
	@echo "  make docs-test          Test all .qmd files (parallel execution)"
	@echo "  make docs-pytest        Run doc tests with pytest (fail-fast)"
	@echo "  make docs-pytest-all    Run all doc tests with pytest (no early exit)"
	@echo "  make docs-test-all      Full test pipeline (jupyter → test → quarto)"
	@echo "  make docs-test-to-debug Test and move fixed notebooks from to_debug/"
	@echo ""
	@echo "Release:"
	@echo "  make pypi-release       Build and upload to PyPI"
	@echo "  make release            Create GitHub release + trigger PyPI publish"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make all                Build + save Docker image"

all: build save

# LinkML schema generation
SCHEMA_PATH = schema/tvbo_datamodel.yaml
DATAMODEL_DIR = tvbo/datamodel
OPENMINDS_DIR = schema/openMINDS_tvbo

gen-linkml:
	@echo "Generating Python datamodel from LinkML schema..."
	@mkdir -p $(DATAMODEL_DIR)
	@gen-pydantic $(SCHEMA_PATH) > $(DATAMODEL_DIR)/pydantic.py
	@gen-python $(SCHEMA_PATH) > $(DATAMODEL_DIR)/schema.py
	@echo "✓ LinkML datamodel generated in $(DATAMODEL_DIR)/"

gen-openminds:
	@echo "Generating openMINDS schemas from LinkML source..."
	@python $(OPENMINDS_DIR)/generate_openminds.py
	@echo "✓ openMINDS schemas generated in $(OPENMINDS_DIR)/schemas/"

OWL_OUT = ontology/tvb-o-struct.owl
SHACL_OUT = ontology/tvb-o.shacl.ttl
ABOX_OUT = ontology/tvb-o-data.ttl

gen-owl:
	@echo "Generating OWL ontology from LinkML schema..."
	@mkdir -p ontology
	@gen-owl $(SCHEMA_PATH) > $(OWL_OUT)
	@python dev/OntologicalRestructuring/tools/postprocess_struct_owl.py --schema $(SCHEMA_PATH) --owl $(OWL_OUT)
	@echo "✓ OWL ontology written to $(OWL_OUT)"

gen-shacl:
	@echo "Generating SHACL shapes from LinkML schema..."
	@mkdir -p ontology
	@gen-shacl $(SCHEMA_PATH) > $(SHACL_OUT)
	@echo "✓ SHACL shapes written to $(SHACL_OUT)"

gen-studies:
	@echo "Converting bibliographies into per-study YAML files..."
	@python dev/OntologicalRestructuring/tools/bib_to_studies.py
	@echo "✓ studies/ regenerated from bibtex"

gen-abox: gen-studies
	@echo "Generating A-box from YAML database..."
	@mkdir -p ontology
	@python dev/OntologicalRestructuring/tools/gen_abox.py -o $(ABOX_OUT)
	@echo "✓ A-box written to $(ABOX_OUT)"

gen-all: gen-linkml gen-openminds gen-owl gen-shacl gen-abox
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

docs-clean:
	@echo "Cleaning generated docs..."
	rm -rf docs/api/ docs/datamodel/ docs/_site/ docs/.quarto/
	rm -f docs/api/.struct_stamp
	@echo "✓ Cleaned: api/, datamodel/, _site/, .quarto/, api/.struct_stamp"

docs-preview:
	@echo "Starting Quarto preview (pre-render runs once, then skipped)..."
	@cd docs && quarto render --no-serve 2>&1 | tail -1
	@cd docs && TVBO_SKIP_PRERENDER=1 quarto preview

docs-render:
	@echo "Full Quarto render..."
	@cd docs && quarto render

docs-publish: docs-render
	@echo "Publishing docs to GitHub Pages..."
	@cd docs && quarto publish gh-pages --no-render --no-prompt

docs-publish-changed:
	@echo "Detecting locally modified .qmd/.md files..."
	@changed=$$(  \
		{  \
			git diff --name-only HEAD 2>/dev/null;  \
			git diff --name-only --cached 2>/dev/null;  \
		} | sort -u | grep '^docs/.*\.\(qmd\|md\)$$'  \
		  | grep -v '^docs/_' | grep -v '__pycache__'  \
	); \
	if [ -z "$$changed" ]; then  \
		echo "No locally modified .qmd/.md files — skipping render.";  \
	else  \
		echo "Rendering $$(echo "$$changed" | wc -l | tr -d ' ') changed file(s):";  \
		echo "$$changed" | sed 's/^/  /';  \
		while IFS= read -r f; do  \
			relpath="$${f#docs/}";  \
			echo "→ quarto render $$relpath";  \
			(cd docs && quarto render "$$relpath") || true;  \
		done <<< "$$changed";  \
	fi
	@echo "Publishing to GitHub Pages..."
	@cd docs && quarto publish gh-pages --no-render --no-prompt

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


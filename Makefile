# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

IMAGE_NAME=tvbo
IMAGE_TAG=latest
IMAGE_FULL=$(IMAGE_NAME):$(IMAGE_TAG)
TARBALL_PATH=/Users/leonmartin_bih/projects/TVB-O/tvbo-container/tvbo.tar.gz

.PHONY: help build save run docs-quarto docs-jupyter docs-to-py docs-rm-py docs-test docs-pytest docs-pytest-all docs-test-all docs-preview docs-preview-guide docs-render docs-render-guide docs-render-api docs-render-datamodel docs-clean docs-publish docs-publish-changed pypi-release release gen-linkml gen-openminds gen-owl gen-shacl gen-neuroml gen-all all check-runtime-onto

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
	@echo "  make docs-test          Test all .qmd files, slow ones included"
	@echo "  make docs-pytest        Run doc tests with pytest (fail-fast)"
	@echo "  make docs-pytest-all    Run all doc tests with pytest (no early exit)"
	@echo "  make docs-test-all      Full test pipeline (jupyter → test → quarto)"
	@echo "  make docs-test-to-debug Test and move fixed notebooks from to_debug/"
	@echo ""
	@echo "Release:"
	@echo "  make pypi-release       Build and upload to PyPI"
	@echo "  make release [BUMP=patch|minor|major | VERSION=x.y.z] [DRYRUN=1]"
	@echo "                          Preview, confirm + publish a GitHub release (auto version bump)"
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
	@# Single source of truth shared with the hatch build hook: hatch_build.py
	@# generates + strips the nondeterministic header, so from-source (`make`) and
	@# build-time (wheel/sdist/editable) codegen produce byte-identical output.
	@python hatch_build.py
	@echo "✓ LinkML datamodel generated in $(DATAMODEL_DIR)/"

gen-openminds:
	@echo "Generating openMINDS schemas from LinkML source..."
	@python $(OPENMINDS_DIR)/generate_openminds.py
	@echo "✓ openMINDS schemas generated in $(OPENMINDS_DIR)/schemas/"

OWL_OUT = ontology/tvb-o-struct.owl
SHACL_OUT = ontology/tvb-o.shacl.ttl
ABOX_OUT = ontology/tvb-o-data.ttl
BIOLOGY_OUT = ontology/tvb-o-biology.ttl
AXIOMS_TTL = ontology/tvb-o-axioms.ttl
BIFURCATION_TTL = ontology/tvb-o-bifurcation.ttl
COUPLING_TTL = ontology/tvb-o-coupling.ttl
NEUROML_TTL = ontology/tvb-o-neuroml.ttl
NEUROML_MAPPINGS = ontology/tvb-o-neuroml-mappings.ttl
NEUROML_CONTRACTS = tvbo/data/ontology/neuroml_contracts.json
UNITS_TTL = ontology/tvb-o-units.ttl
CLINICAL_TTL = ontology/tvb-o-clinical.ttl
CLINICAL_NMM = ontology/tvb-o-clinical-nmm.ttl
MERGED_OUT = ontology/tvbo.owl
# Packaged copy of the generated ontology that the runtime actually loads
# (tvbo/ontology/owl.py). Shipped in the wheel via MANIFEST.in.
RUNTIME_GEN = tvbo/data/ontology/tvbo.owl
# Deprecated class-based ontology — preserved as a parity reference, no longer loaded.
RUNTIME_ONTO = tvbo/data/ontology/tvb-o.owl
WIDOCO_OUT = docs/1-explore/ontology/spec
ROBOT ?= robot
WIDOCO_IMAGE ?= ghcr.io/dgarijo/widoco:v1.4.25

gen-owl:
	@echo "Generating OWL ontology from LinkML schema..."
	@mkdir -p ontology
	@gen-owl $(SCHEMA_PATH) > $(OWL_OUT)
	@python scripts/ontology/postprocess_struct_owl.py --schema $(SCHEMA_PATH) --owl $(OWL_OUT)
	@echo "✓ OWL ontology written to $(OWL_OUT)"

gen-shacl:
	@echo "Generating SHACL shapes from LinkML schema..."
	@mkdir -p ontology
	@gen-shacl $(SCHEMA_PATH) > $(SHACL_OUT)
	@# sh:ignoredProperties denotes a set, and LinkML emits it in Python set order.
	@python scripts/ontology/canonical_ttl.py $(SHACL_OUT) --sort-list sh:ignoredProperties
	@echo "✓ SHACL shapes written to $(SHACL_OUT)"

gen-studies:
	@echo "Converting bibliographies into per-study YAML files..."
	@python scripts/ontology/bib_to_studies.py
	@echo "✓ studies/ regenerated from bibtex"

gen-abox: gen-studies
	@echo "Generating A-box from YAML database..."
	@mkdir -p ontology
	@python scripts/ontology/gen_abox.py -o $(ABOX_OUT) --bio-output $(BIOLOGY_OUT)
	@echo "✓ A-box written to $(ABOX_OUT) (+ biology grounding to $(BIOLOGY_OUT))"

# Ingest the NeuroML2 core LEMS ComponentTypes into a mergeable ontology module
# plus the accumulated contract index the NeuroML adapter loads. Needs the
# neuroml extra (jNeuroML jar + pylems); the committed outputs are consumed by
# gen-merged without regenerating, so a merge does not require the jar.
gen-neuroml:
	@echo "Ingesting NeuroML-core ComponentTypes into the ontology..."
	@mkdir -p ontology
	@python scripts/ontology/gen_neuroml.py -o $(NEUROML_TTL) --contracts $(NEUROML_CONTRACTS)
	@echo "✓ NeuroML module written to $(NEUROML_TTL) (+ contract index $(NEUROML_CONTRACTS))"

crosswalk:
	@echo "Refreshing crosswalk + boundary-matrix from schema/api/odoo..."
	@python scripts/ontology/backfill_crosswalk.py
	@echo "✓ {crosswalk,boundary-matrix}.md updated in $${TVBO_CROSSWALK_DIR:-dev/OntologicalRestructuring}"

# Vendor the QUDT records for every UnitEnum value. Needs network access, so it is
# not part of gen-all; CI checks freshness instead of regenerating.
gen-units:
	@python scripts/ontology/gen_units.py

gen-all: gen-linkml gen-openminds gen-owl gen-shacl gen-abox gen-neuroml gen-merged
	@echo "✓ All schemas generated"

gen-merged: gen-owl gen-abox
	@echo "Merging T-box (struct + axioms) and A-box into a single distributable OWL file..."
	@mkdir -p ontology
	@$(ROBOT) merge \
		--input $(OWL_OUT) \
		--input $(AXIOMS_TTL) \
		--input $(BIFURCATION_TTL) \
		--input $(COUPLING_TTL) \
		--input $(NEUROML_TTL) \
		--input $(NEUROML_MAPPINGS) \
		--input $(UNITS_TTL) \
		--input $(ABOX_OUT) \
		--input $(BIOLOGY_OUT) \
		--input $(CLINICAL_TTL) \
		--input $(CLINICAL_NMM) \
		query --update ontology/fix-punning.ru --update ontology/clinical-postmerge.ru \
		annotate \
		--ontology-iri "https://w3id.org/tvbo/tvbo.owl" \
		--version-iri "https://w3id.org/tvbo/$(shell date +%Y-%m-%d)/tvbo.owl" \
		reason --reasoner ELK \
		--output $(MERGED_OUT)
	@cp $(MERGED_OUT) $(RUNTIME_GEN)
	@echo "✓ Merged ontology written to $(MERGED_OUT) and packaged to $(RUNTIME_GEN)"

# Layer the clinical addon into the ontology artifact that the runtime actually loads
# (tvbo/ontology/owl.py loads RUNTIME_ONTO, not MERGED_OUT). Idempotent: re-merging the
# same triples is a no-op and the label INSERTs are guarded.
gen-runtime-onto:
	@echo "Merging clinical addon into the runtime ontology artifact ($(RUNTIME_ONTO))..."
	@$(ROBOT) merge \
		--input $(RUNTIME_ONTO) \
		--input $(CLINICAL_TTL) \
		--input $(CLINICAL_NMM) \
		query --update ontology/clinical-postmerge.ru \
		--output $(RUNTIME_ONTO)
	@echo "✓ Runtime ontology updated: $(RUNTIME_ONTO)"

# Fail if the runtime ontology the platform KG loads ($(RUNTIME_ONTO)) is older
# than its sources — gen-runtime-onto only re-layers the clinical addon, so a
# struct/axioms/abox edit otherwise never reaches the deployed KG unnoticed.
# Suitable as a CI gate next to the existing "regenerated == committed" checks.
check-runtime-onto:
	@python3 scripts/ontology/check_runtime_onto_fresh.py

gen-widoco: gen-merged
	@echo "Generating Widoco HTML documentation (W3C-style spec + WebVOWL) via Docker..."
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "ERROR: docker not found on PATH. Install Docker or run via CI."; \
		exit 1; \
	fi
	@rm -rf $(WIDOCO_OUT)
	@mkdir -p $(WIDOCO_OUT)
	@docker run --rm --platform linux/amd64 \
		-v "$(PWD)/ontology:/usr/local/widoco/in:ro" \
		-v "$(PWD)/$(WIDOCO_OUT):/usr/local/widoco/out" \
		$(WIDOCO_IMAGE) \
		-ontFile in/tvbo.owl \
		-outFolder out \
		-rewriteAll \
		-webVowl \
		-includeAnnotationProperties \
		-getOntologyMetadata \
		-uniteSections \
		-lang en
	@cp $(MERGED_OUT) $(WIDOCO_OUT)/tvbo.owl
	@cp $(AXIOMS_TTL) $(WIDOCO_OUT)/tvb-o-axioms.ttl
	@cp $(SHACL_OUT) $(WIDOCO_OUT)/tvb-o.shacl.ttl
	@echo "✓ Widoco docs written to $(WIDOCO_OUT)/ (served at /ontology/spec/ by Quarto)"

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
# Docs live in one module, so the default loadscope would pin them all to one worker.
DOCS_PYTEST = pytest tests/test_docs.py -v --tb=short -n $(DOCS_TEST_JOBS) --dist=load

# Every doc, slow ones included; tests/test_docs.py owns discovery and kernel pinning.
docs-test:
	@echo "Testing all .qmd files in docs/ ($(DOCS_TEST_JOBS) parallel jobs)..."
	$(DOCS_PYTEST) --run-slow

# Pytest-based docs testing (for CI/CD)
# Requires: pip install pytest-xdist
docs-pytest:
	@echo "Running documentation tests with pytest ($(DOCS_TEST_JOBS) workers)..."
	$(DOCS_PYTEST) -x

docs-pytest-all:
	@echo "Running all documentation tests ($(DOCS_TEST_JOBS) workers, no early exit)..."
	$(DOCS_PYTEST)

docs-test-all: docs-jupyter docs-test docs-quarto
	@echo "Full test pipeline completed!"

docs-gen-datamodel:
	@echo "Generating LinkML datamodel documentation..."
	@cd docs && python scripts/generate_datamodel_docs.py
	@echo "✓ DataModel documentation generated in docs/datamodel/"

# The docs declare `jupyter: python3`, and that kernelspec's argv is a bare `python`, so PATH decides which interpreter executes the notebooks — on a machine with several project virtualenvs that is silently the wrong one, and the pages then fail against a stale released tvbo. These targets pin the kernel to this checkout's interpreter, the same way tests/test_docs.py does. QUARTO_PYTHON does not cover this; the kernelspec does.
DOCS_VENV := $(CURDIR)/.venv/bin/python
DOCS_KERNEL_DIR := $(CURDIR)/.docs-kernel

$(DOCS_KERNEL_DIR)/kernels/python3/kernel.json:
	@mkdir -p $(DOCS_KERNEL_DIR)/kernels/python3
	@printf '{\n "argv": ["$(DOCS_VENV)", "-m", "ipykernel_launcher", "-f", "{connection_file}"],\n "display_name": "Python 3 (tvbo)",\n "language": "python"\n}\n' > $@
	@echo "✓ docs kernel pinned to $(DOCS_VENV)"

docs-kernel: $(DOCS_KERNEL_DIR)/kernels/python3/kernel.json

# The page-level rules the formatter cannot reach: hard-wrapped prose, the page-style budget, and every documented `tvbo` command resolved against the installed CLI.
DOCS_PYTHON = $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
DOCS_PAGES = $(shell find docs -name '*.qmd' -o -name '*.md' | grep -vE '/(_site|_freeze|_build|\.quarto|_archive|\.jupyter_cache|_output|api|datamodel)/')

docs-lint:
	@python3 ~/.claude/tools/slopfmt.py $(DOCS_PAGES) docs/scripts/*.py docs/filters/*.lua docs/_static/*.yml
	@$(DOCS_PYTHON) docs/scripts/check_pages.py --quiet $(DOCS_PAGES)
	@$(DOCS_PYTHON) docs/scripts/check_cli_examples.py --quiet $(DOCS_PAGES)
	@cd docs && $(abspath $(DOCS_PYTHON)) scripts/check_render_coverage.py
	@cd docs && $(abspath $(DOCS_PYTHON)) scripts/check_citations.py
	@cd docs && $(abspath $(DOCS_PYTHON)) scripts/check_native_pages.py

docs-unwrap:
	@$(DOCS_PYTHON) docs/scripts/unwrap_prose.py $(DOCS_PAGES)

# Not part of docs-lint or CI: it needs the network, and a third-party site being down is not a build failure.
docs-links:
	@$(DOCS_PYTHON) docs/scripts/check_links.py

docs-clean:
	@echo "Cleaning generated docs..."
	rm -rf docs/api/ docs/datamodel/ docs/_site/ docs/.quarto/
	rm -f docs/api/.struct_stamp
	@echo "✓ Cleaned: api/, datamodel/, _site/, .quarto/, api/.struct_stamp"

docs-preview: docs-kernel
	@echo "Starting Quarto preview (pre-render runs once, then skipped)..."
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) quarto render --no-serve 2>&1 | tail -1
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) TVBO_SKIP_PRERENDER=1 quarto preview

docs-render: docs-kernel
	@echo "Full Quarto render..."
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) quarto render

# The three parts of the site render independently. The guide is the authoring loop and skips both generated references; the references are subtrees, so Quarto narrows them by path and the profile only tells the pre-render what to build.
docs-render-guide: docs-kernel
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) QUARTO_PROFILE=guide quarto render

docs-render-api: docs-kernel
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) QUARTO_PROFILE=api quarto render api

docs-render-datamodel: docs-kernel
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) QUARTO_PROFILE=datamodel quarto render datamodel

docs-preview-guide: docs-kernel
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) QUARTO_PROFILE=guide quarto render --no-serve 2>&1 | tail -1
	@cd docs && JUPYTER_PATH=$(DOCS_KERNEL_DIR) QUARTO_PROFILE=guide TVBO_SKIP_PRERENDER=1 quarto preview

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


# Cut a release. Delegates to scripts/release.sh, which previews what is
# shipping, verifies a forward version bump over the latest published release,
# and asks for confirmation before committing/pushing/tagging. Examples:
#   make release BUMP=patch          # auto next patch (x.y.Z+1)
#   make release BUMP=minor          # auto next minor (x.Y+1.0)
#   make release VERSION=0.6.0       # explicit version
#   make release DRYRUN=1            # preview only, change nothing
#   make release                     # release version currently in tvbo/__init__.py
release:
	@VERSION="$(VERSION)" BUMP="$(BUMP)" CONFIRM="$(CONFIRM)" DRYRUN="$(DRYRUN)" bash scripts/release.sh


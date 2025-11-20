# Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

IMAGE_NAME=tvbo
IMAGE_TAG=latest
IMAGE_FULL=$(IMAGE_NAME):$(IMAGE_TAG)
TARBALL_PATH=/Users/leonmartin_bih/projects/TVB-O/tvbo-container/tvbo.tar.gz

.PHONY: build save run docs-quarto docs-jupyter docs-to-py docs-rm-py docs-test docs-test-all all
all: build save

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

docs-test:
	@mkdir -p ./docs/to_debug
	@echo "Testing all .ipynb files in docs/Usage..."
	@echo "========================================"
	@passed=0; failed=0; \
	find ./docs/Usage -name '*.ipynb' -type f | while read notebook; do \
		echo ""; \
		echo "Testing: $$notebook"; \
		if MPLBACKEND=Agg jupyter nbconvert --execute --to notebook --stdout "$$notebook" > /dev/null 2>&1; then \
			echo "✓ PASSED"; \
			passed=$$((passed + 1)); \
		else \
			echo "✗ FAILED"; \
			failed=$$((failed + 1)); \
			relpath=$$(echo "$$notebook" | sed 's|./docs/Usage/||'); \
			targetdir=$$(dirname "./docs/to_debug/$$relpath"); \
			mkdir -p "$$targetdir"; \
			echo "  Moving $$notebook to $$targetdir/"; \
			mv "$$notebook" "$$targetdir/"; \
			qmdfile="$${notebook%.ipynb}.qmd"; \
			if [ -f "$$qmdfile" ]; then \
				echo "  Moving $$qmdfile to $$targetdir/"; \
				mv "$$qmdfile" "$$targetdir/"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "Test Summary:"; \
	echo "  Passed: $$passed"; \
	echo "  Failed: $$failed"; \
	echo "========================================"

docs-test-all: docs-jupyter docs-test docs-quarto
	@echo "Full test pipeline completed!"

docs-gen-datamodel:
	@echo "Generating LinkML datamodel documentation..."
	@cd docs && python scripts/generate_datamodel_docs.py
	@echo "✓ DataModel documentation generated in docs/datamodel/"

docs-preview: docs-gen-datamodel
	@echo "Building Quarto preview with datamodel..."
	@cd docs && quarto preview

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


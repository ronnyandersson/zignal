all: check

.PHONY: check
check: isort flake8 test

.PHONY: isort
isort:
	isort --check-only --diff --gitignore --color \
		--force-grid-wrap 2 --multi-line 8 --trailing-comma src/
	@echo ""

.PHONY: flake8
flake8:
	flake8 --extend-ignore=E265 --statistics src/
	@echo ""

.PHONY: test
test:
	python -m unittest --verbose --buffer src/tests/test_*.py
	@echo ""

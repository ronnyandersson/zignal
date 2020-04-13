all: check

.PHONY: check
check: isort flake8 test

.PHONY: isort
isort:
	isort zignal/*.py zignal/**/*.py examples/*.py --check-only
	@echo ""

.PHONY: flake8
flake8:
	flake8 --extend-ignore=E265 --statistics zignal/ examples/
	@echo ""

.PHONY: test
test:
	nosetests
	@echo ""

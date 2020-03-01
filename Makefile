all: check

.PHONY: check
check: isort flake8

.PHONY: isort
isort:
	@echo "\n*** Running isort, check-only\n"
	isort zignal/**/*.py --check-only
	isort examples/*.py --check-only

.PHONY: flake8
flake8:
	@echo "\n*** Running flake8\n"
	flake8 --extend-ignore=E265 --statistics zignal/

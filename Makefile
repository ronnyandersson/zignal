all: check

.PHONY: check
check: isort flake8

.PHONY: isort
isort:
	@echo ""
	@echo "*** Running isort, check-only"
	@echo ""
	isort zignal/**/*.py --check-only
	isort examples/*.py --check-only

.PHONY: flake8
flake8:
	@echo ""
	@echo "*** Running flake8"
	@echo ""
	flake8 --extend-ignore=E265 --statistics zignal/

.PHONY: help clean build-docker test lint mypy lint dist

# AutoDoc
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


VERSION := $(shell python -c "import sys;sys.path[:0]=('src',);import __meta__ as m;print(m.__version__)")
PROJECT := $(shell python -c "import sys;sys.path[:0]=('src',);import __meta__ as m;print(m.__app__)")

target:
	echo $(VERSION)

help: ## launches this list
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-envs clean-libs clean-build clean-test clean-pyc## remove all build, test, coverage and Python artifacts

clean-envs:
	rm -rf env

clean-libs:
	rm -rf src/libs

clean-build: ## remove build artifacts
	rm -rf build dist
	rm -rf $(find . -name '*.egg-info')

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -fr {} +
	find . -name '*.pyo' -exec rm -fr {} +
	find . -name '*~' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +	 

dist: clean
	python setup.py bdist_egg
	mkdir -p dist/${VERSION}
	cp src/main.py dist/${VERSION} && mv dist/*.egg dist/${VERSION}

deploy-develop: dist ## Build and upload develop egg to S3
	## You can move the artifact to S3 or anywhere else

deploy-master: dist ## Build and upload master egg to S3
	## You can move the artifact to S3 or anywhere else

lint: ## Linting
	flake8 src


test: ## Install and run tests
	python -m pytest -v

clean-test: ## remove test and coverage artifacts
	rm -rf .tox .coverage htmlcov coverage-reports tests.xml tests.html
	rm -rf .coverage.*
	rm -rf .pytest_cache
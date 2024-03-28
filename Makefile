CIRCLE_NODE_INDEX ?= 0
CIRCLE_NODE_TOTAL ?= 1

out=out
out-docker=out-docker
coverage_dir=$(out)/coverage
tr=$(out)/test-results
xunit_output=$(tr)/nose-$(CIRCLE_NODE_INDEX)-xunit.xml


test_packages=dg_commons_tests
cover_packages=dg_commons

junit=--junitxml=$(tr)/junit.xml
parallel=-n auto --dist=loadfile
coverage=--cov-config=pyproject.toml --cov=$(cover_packages) --cov-report html
extra=--capture=no -v
################################

clean:
	coverage erase
	rm -rf $(out) $(coverage_dir) $(tr)

test: clean
	mkdir -p  $(tr)
	DISABLE_CONTRACTS=1 poetry run pytest $(coverage) $(extra) $(junit) src

test-parallel: clean
	mkdir -p  $(tr)
	DISABLE_CONTRACTS=1 poetry run pytest $(coverage) $(extra) $(junit) $(parallel) src

#test-parallel-circle:
#	DISABLE_CONTRACTS=1 \
#	NODE_TOTAL=$(CIRCLE_NODE_TOTAL) \
#	NODE_INDEX=$(CIRCLE_NODE_INDEX) \
#	nosetests $(coverage) $(xunitmp) src -v  $(parallel)

coverage-combine:
	coverage combine

coverage-report:
	coverage html -d $(coverage_dir)

black:
	black -l 120 --target-version py311 src

### Docs ###
docs:
	python -m sphinx.cmd.build src $(out)/docs

docs-docker: build
	mkdir -p $(out-docker)/docs
	docker run -it --rm --user $$(id -u)\
		-v ${PWD}/src:/driving_games/src:ro \
		-v ${PWD}/$(out-docker)/docs:/driving_games/$(out)/docs $(tag) \
		sphinx-build src /driving-games/$(out)/docs

### PyPi versioning ###
include makefiles/Pypi.mk

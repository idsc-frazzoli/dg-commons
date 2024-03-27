
version-publish:
	$(MAKE) version
	$(MAKE) build
	$(MAKE) publish

version:
	poetry version patch
build:
	rm -f dist/*
	rm -rf src/*.egg-info
	poetry build
publish:
	poetry publish

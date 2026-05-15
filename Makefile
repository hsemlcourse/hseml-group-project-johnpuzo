lint:
	flake8 src tests

test:
	pytest

check:
	flake8 src tests
	pytest

notebook:
	jupyter notebook
.PHONY: all tests clean smoke

all:
	python3 run_all.py

smoke:
	python3 run_all.py --smoke

tests:
	python3 -m pytest tests/ -v

clean:
	rm -rf __pycache__ src/*/__pycache__
	rm -rf runs/latest

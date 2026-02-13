.PHONY: deploy test clean

deploy:
	@./scripts/deploy.sh "$(msg)"

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__

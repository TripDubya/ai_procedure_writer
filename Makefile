.PHONY: install test lint run clean docker-build docker-run

install:
    pip install -r requirements/dev.txt

test:
    pytest tests/ -v

lint:
    flake8 src/
    black src/ --check
    isort src/ --check-only

format:
    black src/
    isort src/

run:
    streamlit run src/gui/streamlit_app.py

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete

docker-build:
    docker-compose -f docker/docker-compose.yml build

docker-run:
    docker-compose -f docker/docker-compose.yml up

docker-stop:
    docker-compose -f docker/docker-compose.yml down
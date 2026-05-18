.PHONY: install test run clean lint format

PYTHON = python3
VENV = venv

install: $(VENV)
	$(VENV)/bin/pip install --upgrade pip -q
	$(VENV)/bin/pip install -r requirements.txt -q
	$(VENV)/bin/pip install ruff -q
	@echo "✅ Dependencies installed"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created"

lint:
	$(VENV)/bin/ruff check .
	@echo "✅ Lint passed"

format:
	$(VENV)/bin/ruff format .
	@echo "✅ Formatted"

test:
	$(VENV)/bin/python -c "from main import get_system; s = get_system(); s.initialize(); print('✅ System OK')"

run:
	$(VENV)/bin/streamlit run app.py

run-cli:
	$(VENV)/bin/python main.py

rebuild:
	$(VENV)/bin/python rebuild_and_test.py

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ src/__pycache__ standalone/__pycache__
	rm -rf .ruff_cache
	@echo "✅ Cleaned"

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
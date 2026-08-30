.PHONY: help install verify data flow quality all test api web dev clean

help:
	@echo "  make install   install python dependencies"
	@echo "  make all       full pipeline: data, checks, results  (~40 s)"
	@echo "  make dev       run backend + frontend together for the demo"
	@echo ""
	@echo "  make verify    invariant suite + anti-circularity audit"
	@echo "  make data      generate datasets only"
	@echo "  make flow      bottleneck detection evaluation"
	@echo "  make quality   escape-window evaluation"
	@echo "  make test      pytest"
	@echo "  make api       backend only, on :8000"
	@echo "  make web       frontend only, on :5173"
	@echo "  make clean     delete generated data and results"

install:
	pip install -r requirements.txt

verify:
	python -m dtwin.audit
	python scripts/test_invariants.py

data:
	python run_all.py

all:
	python run_all.py

flow:
	python scripts/eval_flow.py --run data/run_s7

quality:
	python scripts/eval_quality.py

test:
	pytest -q

api:
	python -m uvicorn api.main:app --reload --port 8000

web:
	cd web && npm install && npm run dev

# Backend and frontend together. The frontend proxies /api and /stream to
# :8000, so open http://localhost:5173 only.
dev:
	@test -d data || python run_all.py
	@echo "backend :8000   frontend :5173   -> open http://localhost:5173"
	@python -m uvicorn api.main:app --port 8000 & echo $$! > /tmp/dtwin-api.pid; \
	 cd web && npm install --silent && npm run dev; \
	 kill `cat /tmp/dtwin-api.pid` 2>/dev/null || true

clean:
	rm -rf data results/*.json results/*.md web/dist web/node_modules
	find . -name __pycache__ -type d -exec rm -rf {} +

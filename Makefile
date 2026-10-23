.PHONY: run test ingest eval docker-up docker-down clean

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v tests/

ingest:
	@echo "Ingesting sample documents..."
	@python3 -c '\
import httpx, pathlib; \
client = httpx.Client(base_url="http://localhost:8000"); \
for p in pathlib.Path("data/sample_docs").glob("*.md"): \
    text = p.read_text(); \
    doc_id = f"doc-{p.stem}"; \
    r = client.post("/ingest", json={"doc_id": doc_id, "text": text}); \
    print(f"Ingested {doc_id}: {r.json()}"); \
'

eval:
	@echo "Running Automated Evaluation Harness Benchmark..."
	@curl -s -X POST http://localhost:8000/evaluate | python3 -m json.tool

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

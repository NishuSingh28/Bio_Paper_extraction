# Contributing

Thank you for helping. A few rules keep the project useful and safe.

## Never commit paper data

This repository ships the application only. Do not commit paper text, raw
XML, chunks, vectors, run outputs, Excel files or your list of paper links.
`.gitignore` blocks the usual file types, and CI fails if one gets through.
Before you open a pull request, run `git status` and read the file list.

Do not commit `.env` or any API key. Put secrets in `.env` (see
`.env.example`). The app never writes `api_key` or `ncbi_api_key` into a run
folder.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```

The tests use no network and no model, except the column-routing tests,
which download the small `all-MiniLM-L6-v2` model once.

## Change the code

- Keep each change small and give it a test.
- Match the style of the code around your change.
- If you change what the chunker outputs, increase `CHUNKER_VERSION` in
  `chunking.py`. If you change the BM25 state, increase `BM25_VERSION` in
  `index.py`. If you change the manifest tables, increase `SCHEMA_VERSION` in
  `cache.py`. A new version writes to a new cache namespace, so old entries
  are never read by mistake.
- To add an embedding provider or an LLM backend, see "Extend it" in
  `README.md`.

## Pull requests

1. Make a branch from `main`.
2. Run `pytest`. All tests must pass.
3. Describe what changed and why in the pull request.

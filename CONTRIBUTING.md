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
- To add an embedding provider or an LLM backend, see below.

## Cache internals

```
<cache_dir>/
  manifest.sqlite                                   status, versions, checksums
  raw/<shard>/<source>_<id>.xml.gz                  original downloads
  parsed/<shard>/<paper_key>.json.gz                sections, metadata
  chunks/<chunk_ns>/<shard>/<paper_key>.parquet     text, section, offsets
  bm25/<chunk_ns>/<shard>/<paper_key>.json.gz       keyword index
  vectors/<chunk_ns>__<embed_ns>/vectors.lance      LanceDB, keyed by chunk_id
  quarantine/<time>/...                             entries that verify moved
```

| Layer | Rebuilt when |
| --- | --- |
| raw | Never. A "not found" answer is asked again after 7 days. |
| chunks, BM25 | The paper text, the chunker version or the chunk size changes. |
| vectors | The chunks or the embedding provider, model or revision changes. |

Each file is written to a temporary name and then renamed. The manifest row is
written after the rename. A stopped run leaves each entry complete or missing.

## Add an LLM backend

1. Subclass `llm.BaseLLM` (see `providers.py`).
2. Implement `complete_json(system, user, schema)`. It returns an `LLMResult`.
3. Add the name to `llm.BACKENDS` and a branch in `llm.build_llm`.
4. Set `LLM_BACKEND` in `.env`.

## Add an embedding provider

1. Subclass `embedder.Embedder` and set `provider = "your_name"`.
2. Implement `_load()` (set `self.dim`) and `_encode(texts, batch_size)`.
3. Call `embedder.register_provider("your_name", YourClass)`.
4. Set `EMBEDDING_PROVIDER` in `.env`. The new provider gets its own cache
   namespace.

## Files

| File | Role |
| --- | --- |
| `app.py`, `cli.py` | Streamlit app and command line |
| `column_pipeline.py` | Settings and wiring shared by the app and the CLI |
| `cache.py` | The paper cache, verify, reset |
| `fetch.py`, `chunking.py` | Download from NCBI and Europe PMC; XML to sections and chunks |
| `index.py`, `embedder.py` | BM25, hybrid retrieval, embedding providers |
| `columns.py`, `grouping.py`, `sections.py` | Column specs and routing |
| `extract_columns.py`, `validate.py` | Model calls and checks |
| `runner.py`, `runstore.py`, `export.py` | Batches, run storage, output sheets |
| `llm.py`, `providers.py` | LLM backends |

## Pull requests

1. Make a branch from `main`.
2. Run `pytest`. All tests must pass.
3. Describe what changed and why in the pull request.

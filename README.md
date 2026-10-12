# Paper Column Extractor

Give it a list of papers and the columns you want. It gives you a spreadsheet
with one row per paper. Each value comes with its evidence sentence and a
confidence score.

- **Your columns:** type them in the app. You do not change code.
- **Your papers:** PMIDs, PMCIDs, DOIs, PubMed URLs or an `.xlsx` export.
  The app reads the full text from PubMed Central when it is open access, and
  the abstract when it is not.
- **Each paper is processed once:** a local cache keeps every download and
  embedding. Later runs, with any columns, reuse it.
- **Local and free by default:** Ollama and sentence-transformers. You need no
  API key.

> This repository contains the application only, with no paper data. You are
> responsible for using the papers that you fetch in agreement with their
> licenses and the NCBI usage policies.

## Quickstart

You need Python 3.11–3.13 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/NishuSingh28/Bio_Paper_extraction.git paper-column-extractor
cd paper-column-extractor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # set NCBI_EMAIL; optional: PAPER_CACHE_DIR
ollama pull qwen2.5:7b
streamlit run app.py
```

In the app, do these steps:

1. Type your columns, or load a template.
2. Paste your paper links.
3. Click **Run**.
4. Download the Excel file.

## Command line

```bash
python cli.py run -c templates/rodent_virus_example.json -l my_links.txt
python cli.py run -c my_columns.json -i export.xlsx -s "Included"   # from a workbook
python cli.py runs                                                  # list runs
```

To stop a run, press Ctrl-C. To continue it, run the same command again.

## Your own columns

Only `name` is necessary. Each other field you add makes the checks stricter.

```json
{
  "name": "My review",
  "columns": [
    {"name": "sample size", "type": "integer", "minimum": 0},
    {"name": "country of sampling"},
    {"name": "detection method", "type": "categorical",
     "allowed_values": "PCR, Serology, Other"}
  ]
}
```

Types: `text`, `integer`, `number`, `boolean`, `categorical`, `date`. Other
fields: `description`, `unit`, `maximum`, `pattern`, `section_hint`.

## Keep the cache in Google Drive

The cache is in `~/.paper-extractor/cache` by default. To reuse it on every
computer, put it in Google Drive:

1. Install [Google Drive for desktop](https://www.google.com/drive/download/).
   Set **My Drive syncing options** to **Mirror files**.
2. Set the folder in `.env`:
   ```bash
   PAPER_CACHE_DIR=~/Library/CloudStorage/GoogleDrive-you@example.com/My Drive/paper-cache   # macOS
   PAPER_CACHE_DIR=G:/My Drive/paper-cache                                                  # Windows
   ```
3. Run `python cli.py cache path` to see the folder in use.

Use the cache on **one computer at a time**. Wait until Drive shows "up to
date" before you change computers. **Do not share the cache folder
publicly**, because it holds the text of your papers.

## Cache commands

```bash
python cli.py cache stats     # size of each layer
python cli.py cache verify    # find and repair bad entries
python cli.py cache reset     # delete the cache
```

A change of chunk size or embedding model writes to a new part of the cache.
Vectors from different models are never mixed.

## Docker (optional)

```bash
docker build -t paper-extractor .
docker run --rm -p 8501:8501 --env-file .env -v "$HOME/.paper-extractor:/data" paper-extractor
```

## More

- Settings: `config.yaml`. Overrides and API keys: `.env`.
- To use Claude or OpenAI, set `LLM_BACKEND` and the API key in `.env`.
- Tests: `pip install -r requirements-dev.txt && pytest`.
- Cache internals, new providers and the file map: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. The license covers this code only, not the papers that you fetch.

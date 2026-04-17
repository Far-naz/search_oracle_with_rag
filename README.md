# search_oracle_with_rag

Oracle Advisor Search is a Streamlit app for finding advisors by research topic, name, section, building, email, publications, activities, or press/media mentions. It combines ChromaDB semantic search with BM25 re-ranking and can generate a short natural-language explanation for the best matches when you provide a Gemini API key.

## What is in this project

- `app.py` is the Streamlit UI.
- `advisors/` contains advisor domain models, repository loading, and match output types.
- `search_engines/` contains BM25 and Chroma search engine/index bootstrap logic.
- `helpers/` contains shared helper utilities.
- `explanations/` contains advisor explainability and highlight generation.
- `generators/` contains data generation/enrichment scripts.
- Legacy top-level files (`search_engine.py`, `chroma_index.py`, `advisors_data.py`, etc.) are thin compatibility wrappers that forward to the new package modules.
- `data/[YOURFILE].json` is the advisor dataset used by the app.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Load the app

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

On Windows, you can also double-click `launch_streamlit.bat` or run it from a terminal. The launcher points to `app.py`.

## How it works

When the app starts, it loads the advisor dataset from `data/cbs_employees.json`, builds or refreshes a ChromaDB index in `./chroma_data`, and lets you search with a chat-style query.

The results page shows:

- advisor name, title, and section
- match score
- matched terms and highlighted evidence
- research output, activities, and press/media entries

If you add a Gemini API key in the sidebar, the app can generate a short explanation of why each advisor matches your query.

## Data refresh

The UI includes a button for refreshing advisor data, but that feature depends on `generators/advisor_profile_enricher.py`. If that file is not present in your checkout, the app still runs and the refresh button shows a notice instead of failing.

## Notes

- The ChromaDB persistence folder is created automatically at runtime.
- If you want to reset the local index, delete `./chroma_data` and start the app again.

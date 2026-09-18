# TF-IDF Based Document Retrieval - Virtual Lab

Interactive Streamlit experiment: represent documents and queries with TF-IDF weights
and rank documents by similarity (classical vector space model).

Sections: **Theory** - **Simulation** - **Quiz** - **Report Generation** (PDF export).

## Run locally

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run tfidf_document_retrieval.py
```

Open http://localhost:8501

## Deploy

### Streamlit Community Cloud (free, recommended)
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. "Create app" -> pick the repo/branch -> main file `tfidf_document_retrieval.py`.
4. Deploy. Dependencies are read from `requirements.txt`.

### Hugging Face Spaces (free)
1. Create a Space at https://huggingface.co/new-space with SDK = **Streamlit**.
2. Upload `tfidf_document_retrieval.py`, `requirements.txt` and rename the entry file
   to `app.py` (or set `app_file: tfidf_document_retrieval.py` in the Space README front-matter).

### Docker (Render / Cloud Run / Fly.io / any VPS)
```bash
docker build -t tfidf-vlab .
docker run -p 8501:8501 tfidf-vlab
```
The container reads `$PORT`, so it works unchanged on Render, Cloud Run and Fly.io.

## Notes
- The app writes a generated `lab_report.pdf` into `static/` at runtime; both are gitignored.
- Requires Streamlit >= 1.38 (uses the `width="stretch"` API).

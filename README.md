# TF-IDF Based Document Retrieval — Virtual Lab

Interactive Streamlit experiment: represent documents and queries with TF-IDF weights
and rank documents by similarity (classical vector space model).

**Live app:** https://tfidf-document-retrieval-vlab-7mdrew67jwxqwdj4nbe2yz.streamlit.app/

Sections: **Theory** · **Simulation** · **Quiz** · **Report Generation** (PDF export).

## Run locally

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run tfidf_document_retrieval.py
```

Open http://localhost:8501

## Project files

| File | Purpose |
| --- | --- |
| `tfidf_document_retrieval.py` | The complete Streamlit application |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme / server settings |
| `Dockerfile` | Optional container build |

## Deploy

Deployed on **Streamlit Community Cloud** — push the repo to GitHub, then at
https://share.streamlit.io choose the repo/branch with main file
`tfidf_document_retrieval.py`. Dependencies are installed from `requirements.txt`.

Alternatively, build the container:

```bash
docker build -t tfidf-vlab .
docker run -p 8501:8501 tfidf-vlab
```

The container reads `$PORT`, so it works unchanged on Render, Cloud Run and Fly.io.

## Notes
- The app writes a generated `lab_report.pdf` into `static/` at runtime; both are gitignored.
- Requires Streamlit >= 1.38 (uses the `width="stretch"` API).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tfidf_document_retrieval.py .
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# $PORT is honoured by Render / Cloud Run / Fly; defaults to 8501 locally
CMD streamlit run tfidf_document_retrieval.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0

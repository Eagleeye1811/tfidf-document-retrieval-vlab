"""
Virtual Laboratory Experiment (Streamlit)
=========================================
Experiment  : TF-IDF Based Document Retrieval
Brief       : Represent documents and queries using TF-IDF and rank documents
              based on similarity (vector space model).
Outcome     : Students implement and explore a traditional vector-space
              retrieval system end-to-end.

Structure (4 core sections, mirroring the lab template):
  1. Theory            : Concepts, objectives, procedure, terminology.
  2. Simulation        : Interactive corpus/query/weighting controls, ranking,
                         TF-IDF matrix heatmap, IR metrics, trial logger.
  3. Quiz              : Self-grading conceptual assessment with feedback.
  4. Report Generation : Student info, trials, observations, PDF download.

No custom CSS is used, so Streamlit's native light/dark themes render cleanly.
Dependencies: streamlit, numpy, pandas, plotly, fpdf2
"""

import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "TF-IDF Based Document Retrieval",
    "objectives": [
        "Explain the vector space model and represent documents/queries as weighted term vectors.",
        "Compute term frequency (TF), document frequency (DF) and inverse document frequency (IDF).",
        "Rank a document collection against a free-text query using cosine similarity.",
        "Compare TF and IDF weighting variants and measure their effect on retrieval quality.",
        "Evaluate ranked output using Precision@K, Recall@K and Average Precision."
    ]
}

THEORY_CONTENT = {
    # Rendered as an ordered sequence of blocks: ("md", markdown) or ("latex", formula).
    # Display equations go through st.latex() - a multi-line $$...$$ inside st.markdown()
    # is not closed by KaTeX and swallows all following text.
    "background_blocks": [
        ("md", r"""
### Overview & Principles

Classical (pre-neural) information retrieval treats every document and every query as a
**vector in a high-dimensional term space**. Each dimension of that space is one vocabulary
term; the value along that dimension is a *weight* that says how important the term is for
that document. Retrieval then reduces to a geometry problem: *which document vector points
in most nearly the same direction as the query vector?*

**1. Term Frequency (TF)** — how often term *t* occurs in document *d*. Raw counts favour long
documents and over-reward repetition, so damped variants are common:

- Raw count: $\mathrm{tf}_{t,d} = f_{t,d}$
- Log (sublinear): $\mathrm{tf}_{t,d} = 1 + \log_{10} f_{t,d}$ for $f_{t,d} > 0$
- Binary: $\mathrm{tf}_{t,d} = 1$ if the term occurs, else $0$
- Augmented: $\mathrm{tf}_{t,d} = 0.5 + 0.5 \cdot \dfrac{f_{t,d}}{\max_{t'} f_{t',d}}$

**2. Document Frequency (DF) and IDF** — $\mathrm{df}_t$ is the number of documents containing
*t*. A term appearing in almost every document (e.g. *"the"*, *"data"* in a data-science corpus)
carries little discriminating power, so its weight is damped:

- Standard: $\mathrm{idf}_t = \log_{10}\dfrac{N}{\mathrm{df}_t}$
- Smooth: $\mathrm{idf}_t = \log_{10}\dfrac{1+N}{1+\mathrm{df}_t} + 1$ (never zero, no divide-by-zero)
- Probabilistic: $\mathrm{idf}_t = \max\!\left(0,\ \log_{10}\dfrac{N-\mathrm{df}_t}{\mathrm{df}_t}\right)$

**3. TF-IDF weight** — the product of the two effects:
        """),
        ("latex", r"w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t"),
        ("md", r"""
A weight is large only when a term is *frequent in this document* **and** *rare in the collection*.

**4. Similarity** — with $\vec{q}$ the query vector and $\vec{d}$ a document vector:
        """),
        ("latex", r"\cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}"
                  r"{\lVert \vec{q} \rVert \, \lVert \vec{d} \rVert}"
                  r" = \sum_t \hat{q}_t \hat{d}_t"),
        ("md", r"""
The cosine ignores vector *length*, so a 20-word abstract and a 2000-word article are compared
on **composition** rather than size. An unnormalised dot product, by contrast, is strongly biased
toward long documents — an effect you can switch on and off in the simulation.

### Workflow & System Overview
1. **Text Preprocessing**: lowercase, tokenize, optionally remove stopwords and apply suffix stemming.
2. **Indexing**: build the vocabulary, count $f_{t,d}$, derive $\mathrm{df}_t$ and $\mathrm{idf}_t$.
3. **Weighting**: build the $N \times |V|$ TF-IDF matrix under the selected TF/IDF scheme.
4. **Query Processing**: embed the query in the *same* term space using the *corpus* IDF values.
5. **Scoring & Ranking**: compute similarity of the query against every document and sort descending.
6. **Evaluation**: judge the ranked list with Precision@K, Recall@K and Average Precision.
        """),
    ],
    "procedure": [
        "Step 1: Read the theory, note the TF, IDF and cosine-similarity definitions.",
        "Step 2: Open the Simulation section from the sidebar navigator.",
        "Step 3: Keep the built-in corpus and pick a preset query (presets carry relevance labels, so IR metrics can be scored).",
        "Step 4: Run with the default scheme (Log TF + Standard IDF + cosine) and note the ranking and Precision@K.",
        "Step 5: Switch the IDF scheme to 'None (TF only)' and observe how common words pollute the ranking.",
        "Step 6: Switch similarity to the unnormalised dot product and observe the bias toward longer documents.",
        "Step 7: Toggle stopword removal and stemming; watch the vocabulary size and scores change.",
        "Step 8: Click 'Record Current Trial' after each configuration (log at least 4 distinct trials).",
        "Step 9: Optionally paste your own corpus (one document per line) and query it.",
        "Step 10: Take the Quiz, then open Report Generation, fill in your details and download the PDF."
    ],
    "key_terms": {
        "Corpus (N)": "The collection of documents being searched; N is the number of documents.",
        "Vocabulary (|V|)": "Set of distinct terms after preprocessing; the dimensionality of the vector space.",
        "Term Frequency (tf)": "Weight derived from how often a term occurs inside one document.",
        "Document Frequency (df)": "Number of documents in the corpus that contain the term.",
        "Inverse Doc. Frequency (idf)": "Rarity weight, log(N/df); high for discriminating terms, ~0 for ubiquitous ones.",
        "TF-IDF weight": "tf x idf; high only when a term is frequent locally and rare globally.",
        "Cosine Similarity": "Dot product of L2-normalised vectors; length-independent measure of direction.",
        "Stopwords": "Extremely common function words (the, of, is) usually discarded before indexing.",
        "Stemming": "Suffix stripping that conflates morphological variants (retrieval/retrieving -> retriev).",
        "Precision@K": "Fraction of the top-K retrieved documents that are relevant.",
        "Recall@K": "Fraction of all relevant documents that appear within the top K.",
        "Average Precision (AP)": "Mean of the precision values measured at each relevant document's rank."
    }
}

# --------------------------------------------------------------------------------------
# Built-in benchmark corpus: 13 short documents over 5 topics, with relevance labels
# --------------------------------------------------------------------------------------

BUILTIN_CORPUS = [
    ("D1",  "Information Retrieval",
     "Information retrieval systems rank documents by relevance to a user query "
     "using an inverted index of terms."),
    ("D2",  "Information Retrieval",
     "The vector space model represents documents as weighted term vectors and measures "
     "similarity using the cosine of the angle between the query vector and each document vector."),
    ("D3",  "Information Retrieval",
     "Term frequency inverse document frequency weighting boosts rare informative terms and "
     "discounts very common words that occur in almost every document of the corpus."),
    ("D4",  "Information Retrieval",
     "A search engine crawls web pages, builds an index and returns ranked results for every "
     "search query typed by the user."),
    ("D5",  "Machine Learning",
     "Machine learning models learn statistical patterns from training data in order to make "
     "predictions on unseen examples."),
    ("D6",  "Machine Learning",
     "Neural networks with many hidden layers perform deep learning on large labelled datasets "
     "and are trained with gradient descent."),
    ("D7",  "Machine Learning",
     "Supervised classification algorithms such as naive bayes and support vector machines "
     "assign class labels to feature vectors extracted from the data."),
    ("D8",  "Space Science",
     "The spacecraft entered orbit around Mars after a long interplanetary cruise through deep space."),
    ("D9",  "Space Science",
     "Astronomers observed a distant galaxy with the space telescope to study star formation and "
     "clouds of cosmic dust."),
    ("D10", "Culinary",
     "Boil the pasta in salted water, then toss it with garlic, olive oil and freshly grated cheese."),
    ("D11", "Culinary",
     "A good bread recipe needs flour, water, yeast and salt, kneaded well and baked in a very hot oven."),
    ("D12", "Sports",
     "The football team won the championship match after a late goal in the second half of the game."),
    ("D13", "Sports",
     "The tennis player served an ace to win the final set of the tournament match."),
]

PRESET_QUERIES = {
    "cosine similarity between document vectors": "Information Retrieval",
    "term weighting for a search engine index": "Information Retrieval",
    "how do neural networks learn from training data": "Machine Learning",
    "classification of feature vectors": "Machine Learning",
    "mars orbit spacecraft mission": "Space Science",
    "recipe with flour and water baked in an oven": "Culinary",
    "championship match winning goal": "Sports",
}

STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "around", "as", "at",
    "be", "been", "being", "between", "both", "but", "by", "can", "do", "does", "each",
    "every", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "may", "more", "most", "much", "no", "not", "of", "on", "one",
    "only", "or", "other", "our", "out", "over", "own", "same", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "up", "use", "used", "using",
    "very", "was", "we", "well", "were", "what", "when", "which", "while", "who", "why",
    "will", "with", "would", "you", "your",
}

SIMULATION_CONFIG = {
    "tf_schemes": ["Raw count", "Log (1 + log tf)", "Binary", "Augmented (0.5 + 0.5 tf/max)"],
    "idf_schemes": ["Standard log(N/df)", "Smooth log((1+N)/(1+df)) + 1", "Probabilistic log((N-df)/df)", "None (TF only)"],
    "similarity_modes": ["Cosine (L2 normalised)", "Dot product (unnormalised)"],
    "top_k_min": 1,
    "top_k_max": 10,
    "top_k_default": 5,
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "In the vector space model, what does each dimension of the document vector represent?",
        "options": [
            "A) One distinct term of the vocabulary",
            "B) One document of the corpus",
            "C) One character of the alphabet",
            "D) One position in the inverted index posting list"
        ],
        "answer_index": 0,
        "explanation": "Documents and queries are vectors in |V|-dimensional space, where each axis is a vocabulary term and the value is that term's weight."
    },
    {
        "id": 2,
        "question": "A term appears in every document of the collection (df = N). Using the standard formula idf = log(N/df), what is its IDF?",
        "options": [
            "A) Exactly 1",
            "B) Exactly 0, so the term cannot discriminate between documents",
            "C) Undefined, because log(1) does not exist",
            "D) The largest IDF value in the vocabulary"
        ],
        "answer_index": 1,
        "explanation": "log(N/N) = log(1) = 0. A term present everywhere contributes nothing to distinguishing documents, which is exactly what IDF is designed to encode."
    },
    {
        "id": 3,
        "question": "Why is cosine similarity generally preferred over a raw dot product for ranking?",
        "options": [
            "A) It is faster to compute than the dot product",
            "B) It guarantees every score is an integer",
            "C) It normalises for vector length, so long documents are not unfairly favoured",
            "D) It removes the need to compute IDF at all"
        ],
        "answer_index": 2,
        "explanation": "Dividing by the L2 norms compares direction (term composition) rather than magnitude, removing the length bias that inflates long documents' dot products."
    },
    {
        "id": 4,
        "question": "What is the effect of sublinear (log) TF weighting, tf = 1 + log(f)?",
        "options": [
            "A) It damps the influence of repeated occurrences of the same term",
            "B) It makes term counts grow exponentially with repetition",
            "C) It converts all non-zero counts to exactly 1",
            "D) It is applied to the IDF factor rather than to the term counts"
        ],
        "answer_index": 0,
        "explanation": "A term occurring 20 times is more important than one occurring twice, but not ten times as important; the logarithm compresses that growth."
    },
    {
        "id": 5,
        "question": "The query vector must be built using which IDF values?",
        "options": [
            "A) IDF values computed from the query text alone",
            "B) The same corpus IDF values used for the document vectors",
            "C) IDF values set to 1 for every query term",
            "D) IDF values recomputed after the ranking is produced"
        ],
        "answer_index": 1,
        "explanation": "Query and documents must live in the same weighted space; the IDF statistics always come from the document collection being searched."
    },
    {
        "id": 6,
        "question": "A retrieval run returns 5 documents of which 3 are relevant, and the collection holds 4 relevant documents in total. What are Precision@5 and Recall@5?",
        "options": [
            "A) P = 0.75, R = 0.60",
            "B) P = 0.60, R = 0.75",
            "C) P = 0.60, R = 0.60",
            "D) P = 0.80, R = 0.50"
        ],
        "answer_index": 1,
        "explanation": "Precision@5 = 3/5 = 0.60 (relevant among retrieved); Recall@5 = 3/4 = 0.75 (relevant found among all relevant)."
    },
    {
        "id": 7,
        "question": "Why is the smoothed IDF variant log((1+N)/(1+df)) + 1 often used in practice?",
        "options": [
            "A) It makes every IDF value negative",
            "B) It removes the need for any TF component",
            "C) It avoids division by zero for unseen terms and never collapses a weight to exactly zero",
            "D) It converts the ranking into a binary relevant/non-relevant decision"
        ],
        "answer_index": 2,
        "explanation": "The +1 terms guard against df = 0, and the trailing +1 keeps ubiquitous terms at a small non-zero weight instead of erasing them entirely."
    },
    {
        "id": 8,
        "question": "What is the main effect of removing stopwords before indexing?",
        "options": [
            "A) Vocabulary size shrinks and scores stop being dominated by uninformative function words",
            "B) The corpus size N increases",
            "C) Cosine similarity can no longer be computed",
            "D) All IDF values become identical"
        ],
        "answer_index": 0,
        "explanation": "Stopwords are high-df, low-information terms; removing them shrinks |V| and reduces noise, though IDF already damps them substantially."
    },
    {
        "id": 9,
        "question": "A query term does not occur anywhere in the corpus. What is its contribution to every document score?",
        "options": [
            "A) It raises all document scores equally",
            "B) It contributes zero, because the corresponding document weights are all zero",
            "C) It causes a division-by-zero error in cosine similarity",
            "D) It is assigned the maximum possible IDF and dominates the ranking"
        ],
        "answer_index": 1,
        "explanation": "The product q_t * d_t is zero for every document because d_t = 0 everywhere; out-of-vocabulary query terms simply drop out of the sum."
    },
    {
        "id": 10,
        "question": "Which is a genuine limitation of TF-IDF retrieval compared with modern semantic (embedding) retrieval?",
        "options": [
            "A) It cannot rank documents at all",
            "B) It requires a GPU to compute the weights",
            "C) It is a bag-of-words model: it ignores word order and misses synonyms such as 'car' and 'automobile'",
            "D) It cannot handle collections larger than ten documents"
        ],
        "answer_index": 2,
        "explanation": "TF-IDF matches surface tokens only. Word order is discarded and lexically different but semantically equal terms share no dimension, causing vocabulary mismatch."
    }
]


# ======================================================================================
# 2. SIMULATION ENGINE - TF-IDF INDEXING, SCORING AND EVALUATION
# ======================================================================================

_SUFFIXES = ("ational", "ization", "iveness", "ously", "ment", "ness", "ing", "edly",
             "ies", "ed", "es", "ly", "s")


def simple_stem(token: str) -> str:
    """Very small suffix-stripping stemmer (Porter-like in spirit, not in rigour)."""
    if token == "ies" or len(token) <= 3:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            stem = token[: -len(suf)]
            if suf == "ies":
                stem += "y"
            return stem
    return token


def tokenize(text: str, remove_stopwords: bool, do_stem: bool) -> list:
    """Lowercase -> alphanumeric tokens -> optional stopword removal -> optional stemming."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_stem:
        tokens = [simple_stem(t) for t in tokens]
    return [t for t in tokens if t]


def apply_tf_weight(counts: np.ndarray, scheme: str) -> np.ndarray:
    """Transform a raw count vector/matrix into TF weights under the chosen scheme."""
    counts = counts.astype(float)
    if scheme == "Raw count":
        return counts
    if scheme == "Binary":
        return (counts > 0).astype(float)
    if scheme.startswith("Log"):
        out = np.zeros_like(counts)
        mask = counts > 0
        out[mask] = 1.0 + np.log10(counts[mask])
        return out
    # Augmented
    if counts.ndim == 1:
        mx = counts.max() if counts.max() > 0 else 1.0
        out = np.zeros_like(counts)
        mask = counts > 0
        out[mask] = 0.5 + 0.5 * (counts[mask] / mx)
        return out
    mx = counts.max(axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    out = 0.5 + 0.5 * (counts / mx)
    out[counts == 0] = 0.0
    return out


def compute_idf(df_counts: np.ndarray, n_docs: int, scheme: str) -> np.ndarray:
    """IDF vector for the vocabulary under the chosen scheme (log base 10)."""
    df_counts = df_counts.astype(float)
    if scheme == "None (TF only)":
        return np.ones_like(df_counts)
    if scheme.startswith("Smooth"):
        return np.log10((1.0 + n_docs) / (1.0 + df_counts)) + 1.0
    if scheme.startswith("Probabilistic"):
        safe_df = np.clip(df_counts, 1e-9, None)
        return np.maximum(0.0, np.log10(np.clip(n_docs - df_counts, 1e-9, None) / safe_df))
    # Standard
    safe_df = np.clip(df_counts, 1e-9, None)
    return np.log10(n_docs / safe_df)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a zero-vector guard."""
    if mat.ndim == 1:
        norm = np.linalg.norm(mat)
        return mat / norm if norm > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def evaluate_ranking(ranked_doc_ids: list, relevant_ids: set, k: int) -> dict:
    """Precision@K, Recall@K and Average Precision over the full ranked list."""
    if not relevant_ids:
        return {"precision_at_k": float("nan"), "recall_at_k": float("nan"),
                "avg_precision": float("nan"), "relevant_total": 0}

    top_k = ranked_doc_ids[:k]
    hits_k = sum(1 for d in top_k if d in relevant_ids)
    precision_at_k = hits_k / max(1, len(top_k))
    recall_at_k = hits_k / len(relevant_ids)

    hits, ap_sum = 0, 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            ap_sum += hits / rank
    avg_precision = ap_sum / len(relevant_ids)

    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k,
            "avg_precision": avg_precision, "relevant_total": len(relevant_ids)}


def run_simulation(docs: list, query: str, tf_scheme: str, idf_scheme: str,
                   similarity_mode: str, remove_stopwords: bool, do_stem: bool,
                   top_k: int, relevant_ids: set) -> dict:
    """
    Full TF-IDF retrieval pipeline.

    docs: list of (doc_id, topic, text) tuples.
    Returns every intermediate artefact needed for the plots and the log book.
    """
    doc_ids = [d[0] for d in docs]
    topics = [d[1] for d in docs]
    texts = [d[2] for d in docs]
    n_docs = len(docs)

    # --- Preprocessing & vocabulary ---------------------------------------------------
    doc_tokens = [tokenize(t, remove_stopwords, do_stem) for t in texts]
    query_tokens = tokenize(query, remove_stopwords, do_stem)
    vocab = sorted({t for toks in doc_tokens for t in toks})
    term_index = {t: i for i, t in enumerate(vocab)}
    n_terms = len(vocab)

    if n_terms == 0:
        return {"empty": True, "doc_ids": doc_ids, "vocab": vocab, "query_tokens": query_tokens}

    # --- Raw count matrix (N x |V|) ---------------------------------------------------
    counts = np.zeros((n_docs, n_terms), dtype=float)
    for i, toks in enumerate(doc_tokens):
        for t in toks:
            counts[i, term_index[t]] += 1.0

    df_counts = (counts > 0).sum(axis=0).astype(float)
    idf = compute_idf(df_counts, n_docs, idf_scheme)

    tf_matrix = apply_tf_weight(counts, tf_scheme)
    tfidf_matrix = tf_matrix * idf

    # --- Query vector in the same term space ------------------------------------------
    q_counts = np.zeros(n_terms, dtype=float)
    oov_terms = []
    for t in query_tokens:
        if t in term_index:
            q_counts[term_index[t]] += 1.0
        else:
            oov_terms.append(t)
    q_tf = apply_tf_weight(q_counts, tf_scheme)
    q_vec = q_tf * idf

    # --- Scoring ----------------------------------------------------------------------
    if similarity_mode.startswith("Cosine"):
        doc_space = l2_normalize(tfidf_matrix)
        query_space = l2_normalize(q_vec)
    else:
        doc_space, query_space = tfidf_matrix, q_vec
    scores = doc_space @ query_space

    order = np.argsort(-scores, kind="stable")
    ranked_doc_ids = [doc_ids[i] for i in order]

    ranking_df = pd.DataFrame({
        "Rank": np.arange(1, n_docs + 1),
        "Doc ID": ranked_doc_ids,
        "Topic": [topics[i] for i in order],
        "Score": [round(float(scores[i]), 4) for i in order],
        "Relevant": ["Yes" if doc_ids[i] in relevant_ids else ("No" if relevant_ids else "-")
                     for i in order],
        "Snippet": [texts[i][:90] + ("..." if len(texts[i]) > 90 else "") for i in order],
    })

    metrics = evaluate_ranking(ranked_doc_ids, relevant_ids, top_k)

    # --- Per-query-term diagnostics ----------------------------------------------------
    matched_terms = sorted({t for t in query_tokens if t in term_index})
    term_rows = []
    for t in matched_terms:
        j = term_index[t]
        term_rows.append({
            "Query Term": t,
            "df": int(df_counts[j]),
            "idf": round(float(idf[j]), 4),
            "Query weight": round(float(q_vec[j]), 4),
            "Max doc weight": round(float(tfidf_matrix[:, j].max()), 4),
        })
    query_terms_df = pd.DataFrame(term_rows)

    return {
        "empty": False,
        "doc_ids": doc_ids,
        "topics": topics,
        "texts": texts,
        "doc_tokens": doc_tokens,
        "query_tokens": query_tokens,
        "vocab": vocab,
        "term_index": term_index,
        "counts": counts,
        "df_counts": df_counts,
        "idf": idf,
        "tfidf_matrix": tfidf_matrix,
        "query_vector": q_vec,
        "scores": scores,
        "order": order,
        "ranked_doc_ids": ranked_doc_ids,
        "ranking_df": ranking_df,
        "query_terms_df": query_terms_df,
        "matched_terms": matched_terms,
        "oov_terms": oov_terms,
        "metrics": metrics,
        "top_doc": ranked_doc_ids[0] if ranked_doc_ids else "-",
        "top_score": float(scores[order[0]]) if n_docs else 0.0,
        "vocab_size": n_terms,
        "n_docs": n_docs,
    }


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def _ascii(text) -> str:
    """FPDF core fonts are latin-1; strip anything they cannot encode."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str, last_run: dict) -> bytes:
    """Compiles experiment records into a formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii(EXPERIMENT_CONFIG["title"]), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, _ascii(student_name or "N/A"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, _ascii(student_id or "N/A"), 0)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, _ascii(date_str or datetime.now().strftime("%Y-%m-%d")), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pct = int((quiz_score / quiz_total) * 100) if quiz_total else 0
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({pct}%)", 0)

    pdf.ln(14)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.cell(5, 5, "-", 0)
        pdf.multi_cell(0, 5, _ascii(obj), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Retrieval Trials", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No retrieval trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        cols = list(trials_df.columns)
        col_w = 190.0 / max(1, len(cols))
        max_chars = max(6, int(col_w / 1.6))

        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        for c in cols:
            pdf.cell(col_w, 6, _ascii(str(c))[:max_chars], border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False
        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, _ascii(val_str)[:max_chars], border=1, align="C", fill=fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    # 3. Last retrieval run summary
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Last Retrieval Run", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    if last_run:
        pdf.multi_cell(0, 5, _ascii(
            f"Query: \"{last_run.get('query', '-')}\"  |  Corpus: {last_run.get('corpus', '-')} "
            f"({last_run.get('n_docs', 0)} docs, vocabulary {last_run.get('vocab_size', 0)} terms)"
        ), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 5, _ascii(
            f"Weighting: TF = {last_run.get('tf_scheme', '-')}; IDF = {last_run.get('idf_scheme', '-')}; "
            f"Similarity = {last_run.get('similarity', '-')}; Stopwords removed = {last_run.get('stopwords', '-')}; "
            f"Stemming = {last_run.get('stemming', '-')}"
        ), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Top ranked documents:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for line in last_run.get("top_lines", []):
            pdf.multi_cell(0, 5, _ascii(f"   {line}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No retrieval run executed in this session.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 4. Discussion & Notes
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "4. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "Log-scaled TF combined with standard IDF and cosine normalisation produced the highest "
        "Precision@K across the recorded trials, while disabling IDF allowed common terms to "
        "dominate the ranking."
    )
    pdf.multi_cell(0, 5, _ascii(notes_text))
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Section 1: Theory, background, objectives, procedure, terminology."""
    st.header("Theoretical Framework & Background")
    for kind, content in THEORY_CONTENT["background_blocks"]:
        if kind == "latex":
            st.latex(content)
        else:
            st.markdown(content)

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Worked Micro-Example")
    st.markdown(r"""
Corpus of $N = 3$ documents, term **"retrieval"** occurring in 1 of them, term **"the"** in all 3:

| Term | $f_{t,d}$ | $\mathrm{df}_t$ | $\mathrm{idf}_t = \log_{10}(N/\mathrm{df}_t)$ | $w_{t,d} = (1+\log f) \cdot \mathrm{idf}$ |
|---|---|---|---|---|
| retrieval | 2 | 1 | $\log_{10}(3/1) = 0.477$ | $(1+0.301)\times0.477 = 0.621$ |
| the | 5 | 3 | $\log_{10}(3/3) = 0.000$ | $(1+0.699)\times0.000 = 0.000$ |

Even though *"the"* occurs more than twice as often, its weight collapses to zero: **frequency
alone is not importance**. That asymmetry is the whole point of the IDF factor.
    """)

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology & Variable Reference"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        )
        st.table(var_df)

    with st.expander("Built-in Benchmark Corpus (13 documents, 5 topics)"):
        st.table(pd.DataFrame(BUILTIN_CORPUS, columns=["Doc ID", "Topic", "Document Text"]))


def _resolve_corpus_and_query():
    """Sidebar-free control block returning (docs, query, relevant_ids, corpus_label)."""
    corpus_mode = st.radio(
        "Document collection",
        options=["Built-in benchmark corpus (13 docs, labelled)", "Custom corpus (paste your own)"],
        horizontal=True
    )

    if corpus_mode.startswith("Built-in"):
        docs = list(BUILTIN_CORPUS)
        corpus_label = "Built-in (13 docs)"
        q_col1, q_col2 = st.columns([2, 3])
        with q_col1:
            query_mode = st.radio("Query source", options=["Preset (scored)", "Custom (unscored)"])
        with q_col2:
            if query_mode.startswith("Preset"):
                query = st.selectbox("Select a query", options=list(PRESET_QUERIES.keys()))
            else:
                query = st.text_input("Type your query", value="ranking documents by term weight")
        if query_mode.startswith("Preset"):
            target_topic = PRESET_QUERIES[query]
            relevant_ids = {d[0] for d in docs if d[1] == target_topic}
            st.caption(
                f"Relevance judgement: all **{target_topic}** documents "
                f"({', '.join(sorted(relevant_ids))}) are treated as relevant for this query."
            )
        else:
            relevant_ids = set()
            st.caption("Custom query: no relevance labels available, so IR metrics are not scored.")
    else:
        corpus_label = "Custom"
        default_corpus = (
            "The cat sat on the warm mat near the window.\n"
            "Dogs and cats are common domestic pets in many homes.\n"
            "The stock market closed lower after the interest rate announcement.\n"
            "Investors watched the market rate of return on their bonds closely."
        )
        raw = st.text_area(
            "Paste your corpus - one document per line",
            value=default_corpus,
            height=150
        )
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        docs = [(f"D{i+1}", "Custom", ln) for i, ln in enumerate(lines)]
        query = st.text_input("Query", value="market rate")
        relevant_ids = set()
        if len(docs) < 2:
            st.warning("Provide at least two documents (one per line) for a meaningful ranking.")

    return docs, query, relevant_ids, corpus_label


def render_simulation_section():
    """Section 2: Interactive TF-IDF retrieval sandbox."""
    st.header("Interactive Simulation Sandbox")
    st.info("Configure the corpus, the query and the weighting scheme, then inspect how the "
            "ranking and the retrieval metrics respond.")

    docs, query, relevant_ids, corpus_label = _resolve_corpus_and_query()

    st.divider()
    st.subheader("Weighting & Scoring Parameters")
    cfg = SIMULATION_CONFIG
    c1, c2, c3 = st.columns(3)
    with c1:
        tf_scheme = st.selectbox("TF weighting scheme", options=cfg["tf_schemes"], index=1)
    with c2:
        idf_scheme = st.selectbox("IDF weighting scheme", options=cfg["idf_schemes"], index=0)
    with c3:
        similarity_mode = st.selectbox("Similarity / normalisation", options=cfg["similarity_modes"], index=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
    with c5:
        do_stem = st.checkbox("Apply suffix stemming", value=False)
    with c6:
        k_max = min(cfg["top_k_max"], max(1, len(docs)))
        if k_max <= cfg["top_k_min"]:
            top_k = cfg["top_k_min"]
            st.caption(f"Top-K fixed at {top_k} (the corpus holds a single document).")
        else:
            top_k = st.slider("Top-K (cut-off for metrics)", min_value=cfg["top_k_min"],
                              max_value=k_max, value=min(cfg["top_k_default"], k_max))

    if len(docs) < 1 or not str(query).strip():
        st.warning("A non-empty corpus and query are required to run the retrieval engine.")
        return

    res = run_simulation(docs, query, tf_scheme, idf_scheme, similarity_mode,
                         remove_stopwords, do_stem, top_k, relevant_ids)

    if res.get("empty"):
        st.error("The vocabulary is empty after preprocessing. Add more text or disable stopword removal.")
        return

    st.divider()

    # ---------------- Metrics ----------------
    m = res["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top-1 Document", res["top_doc"], help="Highest scoring document")
    with m2:
        st.metric("Top Score", f"{res['top_score']:.4f}")
    with m3:
        st.metric(f"Precision@{top_k}",
                  "-" if math.isnan(m["precision_at_k"]) else f"{m['precision_at_k']:.2f}")
    with m4:
        st.metric("Average Precision",
                  "-" if math.isnan(m["avg_precision"]) else f"{m['avg_precision']:.3f}")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("Documents (N)", res["n_docs"])
    with m6:
        st.metric("Vocabulary |V|", res["vocab_size"])
    with m7:
        st.metric("Matched query terms", len(res["matched_terms"]))
    with m8:
        st.metric(f"Recall@{top_k}",
                  "-" if math.isnan(m["recall_at_k"]) else f"{m['recall_at_k']:.2f}")

    if res["oov_terms"]:
        st.caption(f"Out-of-vocabulary query terms (contribute nothing to any score): "
                   f"`{', '.join(sorted(set(res['oov_terms'])))}`")

    # ---------------- Ranked results ----------------
    st.subheader("Ranked Retrieval Results")
    st.dataframe(res["ranking_df"].head(max(top_k, 5)), width="stretch", hide_index=True)

    # ---------------- Score bar chart ----------------
    st.subheader("Similarity Score Distribution")
    rk = res["ranking_df"]
    colors = ["#2563eb" if r == "Yes" else ("#94a3b8" if r == "No" else "#2563eb")
              for r in rk["Relevant"]]
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        x=rk["Score"][::-1], y=rk["Doc ID"][::-1], orientation="h",
        marker=dict(color=colors[::-1]),
        hovertext=rk["Snippet"][::-1], name="Similarity"
    ))
    fig_scores.update_layout(
        title=f"Query: \"{query}\"  -  blue = relevant / unlabelled, grey = non-relevant",
        xaxis_title="Similarity score", yaxis_title="Document",
        height=max(320, 26 * len(rk)), margin=dict(l=20, r=20, t=50, b=20), showlegend=False
    )
    st.plotly_chart(fig_scores, width="stretch")

    # ---------------- Query term diagnostics ----------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Query Term Weights")
        if res["query_terms_df"].empty:
            st.info("No query term occurs in the corpus vocabulary.")
        else:
            st.dataframe(res["query_terms_df"], width="stretch", hide_index=True)

    with col_b:
        st.subheader("IDF vs Document Frequency")
        df_counts, idf = res["df_counts"], res["idf"]
        fig_idf = go.Figure()
        fig_idf.add_trace(go.Scatter(
            x=df_counts, y=idf, mode="markers", name="Vocabulary terms",
            text=res["vocab"], marker=dict(size=7, opacity=0.6)
        ))
        if res["matched_terms"]:
            idxs = [res["term_index"][t] for t in res["matched_terms"]]
            fig_idf.add_trace(go.Scatter(
                x=df_counts[idxs], y=idf[idxs], mode="markers+text", name="Query terms",
                text=res["matched_terms"], textposition="top center",
                marker=dict(size=12, symbol="diamond", color="#ef4444")
            ))
        fig_idf.update_layout(
            title=f"IDF decay ({idf_scheme})", xaxis_title="Document frequency (df)",
            yaxis_title="IDF weight", height=380, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_idf, width="stretch")

    # ---------------- TF-IDF matrix heatmap ----------------
    st.subheader("TF-IDF Weight Matrix (most informative terms)")
    mat = res["tfidf_matrix"]
    term_cap = min(30, res["vocab_size"])
    if term_cap <= 5:
        max_terms = term_cap
    else:
        max_terms = st.slider("Number of terms to display", 5, term_cap,
                              min(14, term_cap))
    top_term_idx = np.argsort(-mat.max(axis=0))[:max_terms]
    term_labels = [res["vocab"][j] for j in top_term_idx]
    heat = mat[:, top_term_idx]
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat, x=term_labels, y=res["doc_ids"], colorscale="Blues",
        colorbar=dict(title="w(t,d)"), hovertemplate="Doc %{y}<br>Term %{x}<br>weight %{z:.3f}<extra></extra>"
    ))
    fig_heat.update_layout(
        title=f"TF ({tf_scheme}) x IDF ({idf_scheme}) weights",
        xaxis_title="Term", yaxis_title="Document",
        height=max(340, 24 * res["n_docs"]), margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_heat, width="stretch")

    # ---------------- Step-by-step computation ----------------
    with st.expander("Step-by-step computation for a single (term, document) pair"):
        if res["matched_terms"]:
            sc1, sc2 = st.columns(2)
            with sc1:
                sel_term = st.selectbox("Term", options=res["matched_terms"])
            with sc2:
                sel_doc = st.selectbox("Document", options=res["doc_ids"],
                                       index=res["doc_ids"].index(res["top_doc"]))
            j = res["term_index"][sel_term]
            i = res["doc_ids"].index(sel_doc)
            raw_f = res["counts"][i, j]
            tf_w = apply_tf_weight(res["counts"], tf_scheme)[i, j]
            idf_w = res["idf"][j]
            w = res["tfidf_matrix"][i, j]
            norm = np.linalg.norm(res["tfidf_matrix"][i])
            st.markdown(f"""
- Raw count in {sel_doc}: **f = {raw_f:.0f}**
- TF weight ({tf_scheme}): **tf = {tf_w:.4f}**
- Document frequency: **df = {int(res['df_counts'][j])}** out of N = {res['n_docs']}
- IDF ({idf_scheme}): **idf = {idf_w:.4f}**
- TF-IDF weight: **w = tf x idf = {tf_w:.4f} x {idf_w:.4f} = {w:.4f}**
- Document vector L2 norm: **||d|| = {norm:.4f}**
- Cosine-normalised component: **w / ||d|| = {(w / norm if norm else 0):.4f}**
- Contribution to the score of {sel_doc}: this component multiplied by the corresponding
  normalised query weight (**{res['query_vector'][j]:.4f}** before normalisation).
            """)
        else:
            st.info("No query term matched the vocabulary, so there is nothing to decompose.")

    # ---------------- Trial logger ----------------
    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current configuration and retrieval metrics into your trial table:")
        if st.button("Record Current Trial", type="primary", width="stretch"):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": (query[:26] + "...") if len(query) > 26 else query,
                "TF": tf_scheme.split(" (")[0],
                "IDF": idf_scheme.split(" ")[0],
                "Sim": "Cosine" if similarity_mode.startswith("Cosine") else "Dot",
                "Stop/Stem": f"{'Y' if remove_stopwords else 'N'}/{'Y' if do_stem else 'N'}",
                "|V|": res["vocab_size"],
                "Top Doc": res["top_doc"],
                "Top Score": round(res["top_score"], 4),
                f"P@K": (None if math.isnan(m["precision_at_k"]) else round(m["precision_at_k"], 3)),
                "AP": (None if math.isnan(m["avg_precision"]) else round(m["avg_precision"], 3)),
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", width="stretch"):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, width="stretch", hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="tfidf_retrieval_trials.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting data.")

    # Persist the last run so the report can summarise it
    top_lines = [
        f"{row['Rank']}. {row['Doc ID']} (score {row['Score']:.4f}, relevant: {row['Relevant']}) - {row['Snippet']}"
        for _, row in res["ranking_df"].head(5).iterrows()
    ]
    st.session_state["last_run"] = {
        "query": query,
        "corpus": corpus_label,
        "n_docs": res["n_docs"],
        "vocab_size": res["vocab_size"],
        "tf_scheme": tf_scheme,
        "idf_scheme": idf_scheme,
        "similarity": similarity_mode,
        "stopwords": "Yes" if remove_stopwords else "No",
        "stemming": "Yes" if do_stem else "No",
        "top_lines": top_lines,
    }


def render_quiz_section():
    """Section 3: Assessment quiz with self-grading and feedback."""
    st.header("Concept Assessment Quiz")
    st.write("Answer the conceptual questions below to evaluate your understanding of TF-IDF retrieval.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: "
                   f"**{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Section 4: Lab report generator with PDF export."""
    st.header("Report Generation")
    st.write("Compile your student details, recorded trials and quiz evaluation into a PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-001"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations and conclusions:",
        value=st.session_state.get("student_notes", (
            "Across the recorded trials, log-scaled TF with standard IDF and cosine normalisation "
            "gave the highest Precision@K. Disabling IDF allowed high-frequency generic terms to "
            "dominate, and the unnormalised dot product biased the ranking toward longer documents."
        )),
        height=140
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()
    last_run = st.session_state.get("last_run", {})

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if last_run:
        st.write(f"**Last run:** query \"{last_run['query']}\" on {last_run['corpus']} "
                 f"using {last_run['tf_scheme']} TF + {last_run['idf_scheme']} IDF, {last_run['similarity']}.")
    else:
        st.info("No retrieval run executed yet - visit the Simulation section first.")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.info("Note: no trials recorded in the Simulation tab yet; the report will show 0 trials.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        last_run=last_run
    )

    # Save to local files for a guaranteed download path
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            width="stretch"
        )
    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            width="stretch"
        )


# ======================================================================================
# 5. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initialises Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = {}


def main():
    st.set_page_config(
        page_title="TF-IDF Document Retrieval | Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption("Vector space model - represent documents and queries with TF-IDF weights and "
               "rank by similarity.")

    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    st.sidebar.write(f"- **Trials Recorded:** `{len(st.session_state['trials'])}`")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()

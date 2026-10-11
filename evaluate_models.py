"""Compare TF-IDF and BM25 retrieval on the CISI collection.

Run this file from the repository root or from the ``week 7`` folder:
    python evaluate_models.py

It reads the Assignment 6 CISI data, evaluates both models, and writes
report-ready CSV files and charts in ``week 7/output``.
"""

from collections import Counter, defaultdict
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem import PorterStemmer

    STEMMER = PorterStemmer()
except ImportError:
    # The evaluation still runs if NLTK is unavailable; it simply skips stemming.
    STEMMER = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "week 6"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
K = 10


def parse_cisi_records(path):
    """Read a CISI .ALL or .QRY file into {record_id: {field: text}}."""
    records = {}
    current_id = None
    current_field = None

    with path.open(encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if line.startswith(".I "):
                current_id = int(line.split()[1])
                records[current_id] = {"T": [], "A": [], "W": [], "X": []}
                current_field = None
            elif line in {".T", ".A", ".W", ".X"}:
                current_field = line[1]
            elif current_id is not None and current_field is not None:
                records[current_id][current_field].append(line.strip())

    for fields in records.values():
        for field, value in fields.items():
            fields[field] = " ".join(value).strip()
    return records


def parse_relevance(path):
    """Read CISI relevance judgments as {query_id: {relevant_document_ids}}."""
    relevance = defaultdict(set)
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            values = line.split()
            if len(values) >= 2:
                relevance[int(values[0])].add(int(values[1]))
    return dict(relevance)


STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess(text):
    """Apply the same normalization to documents and queries."""
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    if STEMMER is not None:
        tokens = [STEMMER.stem(token) for token in tokens]
    return tokens


def precision_at_k(ranked_ids, relevant_ids, k):
    return sum(doc_id in relevant_ids for doc_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return np.nan
    return sum(doc_id in relevant_ids for doc_id in ranked_ids[:k]) / len(relevant_ids)


def average_precision(ranked_ids, relevant_ids):
    """Compute AP over the full ranked list; MAP is the mean AP across queries."""
    if not relevant_ids:
        return np.nan

    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / rank
    return total / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevant_ids, k):
    """Compute binary nDCG@k: 1 for relevant, 0 for non-relevant documents."""
    dcg = sum(
        (1 / math.log2(rank + 1))
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else np.nan


def bm25_scores(query_tokens, document_term_counts, document_lengths, document_frequencies, average_length):
    """Return BM25 scores for one query, using standard k1 and b settings."""
    k1, b = 1.5, 0.75
    total_documents = len(document_term_counts)
    scores = np.zeros(total_documents)
    query_terms = set(query_tokens)

    for index, term_counts in enumerate(document_term_counts):
        length_normalizer = k1 * (1 - b + b * document_lengths[index] / average_length)
        for term in query_terms:
            frequency = term_counts[term]
            if not frequency:
                continue
            # Add 0.5 smoothing to the inverse-document-frequency calculation.
            idf = math.log((total_documents - document_frequencies[term] + 0.5) /
                           (document_frequencies[term] + 0.5) + 1)
            scores[index] += idf * frequency * (k1 + 1) / (frequency + length_normalizer)
    return scores


def evaluate_model(model_name, rank_query, query_ids, queries, relevance):
    """Evaluate every judged query and return one row per query."""
    rows = []
    for query_id in query_ids:
        ranked_ids = rank_query(queries[query_id]["W"])
        relevant_ids = relevance[query_id]
        precision = precision_at_k(ranked_ids, relevant_ids, K)
        recall = recall_at_k(ranked_ids, relevant_ids, K)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "Model": model_name,
            "Query_ID": query_id,
            f"Precision@{K}": precision,
            f"Recall@{K}": recall,
            f"F1@{K}": f1,
            "Average_Precision": average_precision(ranked_ids, relevant_ids),
            f"nDCG@{K}": ndcg_at_k(ranked_ids, relevant_ids, K),
            "Relevant_Documents": len(relevant_ids),
        })
    return rows


def save_charts(summary):
    """Create charts that can be inserted directly into the written report."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    chart_columns = [f"Precision@{K}", f"Recall@{K}", f"F1@{K}", "MAP", f"nDCG@{K}"]
    positions = np.arange(len(chart_columns))
    width = 0.35

    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, (_, row) in enumerate(summary.iterrows()):
        axis.bar(positions + (offset - 0.5) * width, row[chart_columns], width, label=row["Model"])
    axis.set_xticks(positions)
    axis.set_xticklabels(chart_columns)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("CISI Retrieval Model Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)


def save_summary_image(summary):
    """Save the CSV summary as a legible image for the written report."""
    display_summary = summary.copy()
    metric_columns = [f"Precision@{K}", f"Recall@{K}", f"F1@{K}", "MAP", f"nDCG@{K}"]
    for column in metric_columns:
        display_summary[column] = display_summary[column].map(lambda value: f"{value:.4f}")

    fig, axis = plt.subplots(figsize=(10, 2.2))
    axis.axis("off")
    table = axis.table(
        cellText=display_summary.values,
        colLabels=display_summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F1F5F9")
    fig.tight_layout(pad=0.35)
    fig.savefig(OUTPUT_DIR / "overall_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    documents_raw = parse_cisi_records(DATA_DIR / "CISI.ALL")
    queries = parse_cisi_records(DATA_DIR / "CISI.QRY")
    relevance = parse_relevance(DATA_DIR / "CISI.REL")
    document_ids = sorted(documents_raw)
    query_ids = sorted(set(queries).intersection(relevance))

    # Include both title and abstract to match the Assignment 6 retrieval project.
    document_tokens = [
        preprocess(f"{documents_raw[doc_id]['T']} {documents_raw[doc_id]['W']}")
        for doc_id in document_ids
    ]
    processed_documents = [" ".join(tokens) for tokens in document_tokens]

    # TF-IDF model: L2-normalized vectors compared with cosine similarity.
    vectorizer = TfidfVectorizer(
        lowercase=False, tokenizer=str.split, preprocessor=None, token_pattern=None, norm="l2"
    )
    tfidf_matrix = vectorizer.fit_transform(processed_documents)

    def rank_tfidf(query):
        query_vector = vectorizer.transform([" ".join(preprocess(query))])
        scores = cosine_similarity(query_vector, tfidf_matrix).ravel()
        return [document_ids[index] for index in np.argsort(-scores)]

    # BM25 model: calculate collection statistics once, then score each query.
    document_frequencies = Counter()
    document_term_counts = []
    document_lengths = []
    for tokens in document_tokens:
        document_frequencies.update(set(tokens))
        document_term_counts.append(Counter(tokens))
        document_lengths.append(len(tokens))
    average_length = sum(map(len, document_tokens)) / len(document_tokens)

    def rank_bm25(query):
        scores = bm25_scores(
            preprocess(query),
            document_term_counts,
            document_lengths,
            document_frequencies,
            average_length,
        )
        return [document_ids[index] for index in np.argsort(-scores)]

    evaluation_rows = []
    evaluation_rows.extend(evaluate_model("TF-IDF", rank_tfidf, query_ids, queries, relevance))
    evaluation_rows.extend(evaluate_model("BM25", rank_bm25, query_ids, queries, relevance))
    per_query = pd.DataFrame(evaluation_rows)

    summary = per_query.groupby("Model", as_index=False).agg({
        f"Precision@{K}": "mean",
        f"Recall@{K}": "mean",
        f"F1@{K}": "mean",
        "Average_Precision": "mean",
        f"nDCG@{K}": "mean",
    }).rename(columns={"Average_Precision": "MAP"})

    per_query.to_csv(OUTPUT_DIR / "per_query_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    save_charts(summary)
    save_summary_image(summary)

    print(f"Evaluated {len(query_ids)} queries against {len(document_ids)} documents.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results and chart in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path
import math
import re
from collections import Counter

import pandas as pd
from rank_bm25 import BM25Okapi

DOCUMENT_FOLDER = Path("documents")
OUTPUT_FOLDER = Path("results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

K1 = 1.5
B = 0.75
LAMBDA = 0.7


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def load_documents():
    files = sorted(DOCUMENT_FOLDER.glob("*.txt"))

    if len(files) < 5:
        raise ValueError(
            "Please place at least 5 .txt documents inside the documents folder."
        )

    names, texts, tokens = [], [], []

    for file in files:
        text = file.read_text(encoding="utf-8")
        names.append(file.stem)
        texts.append(text)
        tokens.append(tokenize(text))

    return files, names, texts, tokens


def calculate_bm25(document_tokens, query_tokens):
    model = BM25Okapi(document_tokens, k1=K1, b=B)
    return model.get_scores(query_tokens)


def jelinek_mercer_score(
    query_tokens,
    document_tokens,
    collection_counts,
    collection_length,
    lambda_value=LAMBDA
):
    if not document_tokens or collection_length == 0:
        return float("-inf")

    document_counts = Counter(document_tokens)
    document_length = len(document_tokens)
    query_counts = Counter(query_tokens)

    score = 0.0

    for term, query_frequency in query_counts.items():
        p_document = document_counts.get(term, 0) / document_length
        p_collection = collection_counts.get(term, 0) / collection_length

        p_smoothed = (
            lambda_value * p_document
            + (1 - lambda_value) * p_collection
        )

        if p_smoothed <= 0:
            return float("-inf")

        score += query_frequency * math.log(p_smoothed)

    return score


def calculate_jm(document_tokens, query_tokens):
    collection = [
        token
        for document in document_tokens
        for token in document
    ]

    collection_counts = Counter(collection)
    collection_length = len(collection)

    return [
        jelinek_mercer_score(
            query_tokens,
            document,
            collection_counts,
            collection_length
        )
        for document in document_tokens
    ]


def save_results(names, bm25_scores, jm_scores):
    bm25 = pd.DataFrame({
        "Document": names,
        "BM25 Score": bm25_scores
    }).sort_values("BM25 Score", ascending=False).reset_index(drop=True)

    bm25["BM25 Rank"] = range(1, len(bm25) + 1)

    jm = pd.DataFrame({
        "Document": names,
        "JM Score": jm_scores
    }).sort_values("JM Score", ascending=False).reset_index(drop=True)

    jm["JM Rank"] = range(1, len(jm) + 1)

    comparison = pd.merge(
        bm25,
        jm,
        on="Document"
    )[[
        "Document",
        "BM25 Rank",
        "BM25 Score",
        "JM Rank",
        "JM Score"
    ]]

    bm25.to_csv(OUTPUT_FOLDER / "bm25_results.csv", index=False)
    jm.to_csv(OUTPUT_FOLDER / "jelinek_mercer_results.csv", index=False)
    comparison.to_csv(OUTPUT_FOLDER / "ranking_comparison.csv", index=False)

    return bm25, jm, comparison


def main():
    _, names, _, document_tokens = load_documents()

    query = input(
        "Enter a search query (example: machine learning): "
    ).strip()

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("The query cannot be empty.")

    bm25_scores = calculate_bm25(document_tokens, query_tokens)
    jm_scores = calculate_jm(document_tokens, query_tokens)

    bm25, jm, comparison = save_results(
        names,
        bm25_scores,
        jm_scores
    )

    print("\n" + "=" * 70)
    print("DOCUMENT RANKING: BM25 vs JELINEK-MERCER")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"BM25: k1={K1}, b={B}")
    print(f"Jelinek-Mercer: lambda={LAMBDA}")

    print("\nOKAPI BM25 RANKING")
    print("-" * 70)

    for _, row in bm25.iterrows():
        print(
            f"{int(row['BM25 Rank'])}. "
            f"{row['Document']:<25} "
            f"{row['BM25 Score']:.6f}"
        )

    print("\nJELINEK-MERCER RANKING")
    print("-" * 70)

    for _, row in jm.iterrows():
        print(
            f"{int(row['JM Rank'])}. "
            f"{row['Document']:<25} "
            f"{row['JM Score']:.6f}"
        )

    print("\nCOMPARISON")
    print("-" * 70)
    print(comparison.to_string(index=False))

    print("\nCSV results saved in the 'results' folder.")


if __name__ == "__main__":
    main()

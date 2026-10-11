# CISI Information Retrieval System

A basic Information Retrieval system implemented in Python using the CISI dataset.

## Retrieval Pipeline
1. Parse CISI documents, queries, and relevance judgments.
2. Lowercase and tokenize text.
3. Remove English stopwords.
4. Apply Porter stemming.
5. Build a TF-IDF document-term matrix.
6. Transform user queries using the same TF-IDF model.
7. Rank documents using cosine similarity.
8. Evaluate with Precision@10, Recall@10, F1@10, Average Precision, and MAP.

## Dataset Statistics
- Documents: 1460
- Queries: 112
- Queries with relevance judgments: 76
- TF-IDF vocabulary size: 5543

## Results from the supplied dataset
- Mean Precision@10: 0.3618
- Mean Recall@10: 0.1383
- Mean F1@10: 0.1756
- MAP: 0.2333

## Files
- `CISI_Information_Retrieval_System.ipynb` — complete implementation
- `CISI.ALL`, `CISI.QRY`, `CISI.REL` — CISI data files
- `cisi_evaluation_results.csv` — per-query evaluation
- `requirements.txt` — Python dependencies

## Run
Place the three CISI data files in the same folder as the notebook and run all cells in Jupyter Notebook or Google Colab.

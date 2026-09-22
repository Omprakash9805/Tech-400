# Document Similarity Using the Vector Space Model

This project compares eight publicly accessible literary sources using TF-IDF and cosine similarity.

## Structure
- `documents/` — eight TXT source-synopsis documents
- `similarity_analysis.py` — reproducible Python code
- `similarity_scores.csv` — ranked pairwise scores
- `similarity_matrix.csv` — complete matrix
- `screenshots/` — code and results snapshots
- `Vector_Space_Model_Report.pdf` — assignment report

## Run
```bash
pip install -r requirements.txt
python similarity_analysis.py
```

The local TXT files are concise original synopses of public Project Gutenberg sources, with source URLs included in each file. This avoids redistributing long source texts while keeping the experiment reproducible.

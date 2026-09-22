from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOC_DIR = Path("documents")
files = sorted(DOC_DIR.glob("*.txt"))

names = [f.stem for f in files]
texts = [f.read_text(encoding="utf-8") for f in files]

# Vector Space Model: TF-IDF representation
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(texts)

# Pairwise cosine similarity
similarity_matrix = cosine_similarity(X)

pd.DataFrame(
    similarity_matrix, index=names, columns=names
).to_csv("similarity_matrix.csv")

pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pairs.append({
            "Document 1": names[i],
            "Document 2": names[j],
            "Similarity": similarity_matrix[i, j]
        })

results = pd.DataFrame(pairs).sort_values(
    "Similarity", ascending=False
)
results.to_csv("similarity_scores.csv", index=False)

print("TF-IDF matrix shape:", X.shape)
print(results.head(10).to_string(index=False))

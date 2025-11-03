# streamlit_app/utils/features.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # streamlit_app/utils
DATA_PATH = os.path.join(BASE_DIR, '../../data/extracted_content.csv')  # project_root/data
MODEL_DIR = os.path.join(BASE_DIR, '../models')               # streamlit_app/models
MODEL_PATH = os.path.join(MODEL_DIR, 'quality_model.pkl')

# ---------------- Load dataset ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ---------------- Clean text ----------------
df['clean_text'] = df['body_text'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

# ---------------- TF-IDF vectorization ----------------
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# Optional: save top keywords for reference
feature_names = vectorizer.get_feature_names_out()
def get_top_k_keywords(row_tfidf, k=5):
    if row_tfidf.nnz == 0:
        return ''
    row = row_tfidf.toarray().flatten()
    top_idx = np.argsort(row)[-k:][::-1]
    top_words = [feature_names[i] for i in top_idx if row[i] > 0]
    return '|'.join(top_words)

df['top_keywords'] = [get_top_k_keywords(tfidf_matrix[i], k=5) for i in range(tfidf_matrix.shape[0])]

# ---------------- Feature columns for model ----------------
feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease']

# ---------------- Optional: Save TF-IDF into model artifacts ----------------
os.makedirs(MODEL_DIR, exist_ok=True)

if os.path.exists(MODEL_PATH):
    try:
        artifacts = joblib.load(MODEL_PATH)
    except Exception:
        artifacts = {}
else:
    artifacts = {}

# Update artifacts with TF-IDF objects (overwrites/sets them)
artifacts.update({
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'feature_cols': feature_cols
})

# Persist artifacts to the new models path
joblib.dump(artifacts, MODEL_PATH)
print(f"Saved TF-IDF artifacts to {MODEL_PATH}")

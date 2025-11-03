# streamlit_app/utils/features.py
import os
import re
import joblib
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # streamlit_app/utils
DATA_PATH = os.path.join(BASE_DIR, '../../data/extracted_content.csv')  # project_root/data
MODEL_DIR = os.path.join(BASE_DIR, '../models')               # streamlit_app/models
MODEL_PATH = os.path.join(MODEL_DIR, 'quality_model.pkl')

# ---------------- Google Drive settings ----------------
DRIVE_FILE_ID = "1YF7BhTtTZeRab_oltPMANK4dBLhgzJ8C"  # from your share link

# ---------------- Helper: download file from Google Drive (handles confirm token) ----------------
def _get_confirm_token(resp):
    # check cookies first
    for key, val in resp.cookies.items():
        if key.startswith('download_warning'):
            return val
    # fallback: search html for confirm token
    try:
        txt = resp.content.decode('utf-8', errors='ignore')
        m = re.search(r"confirm=([0-9A-Za-z_-]+)&", txt)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def download_from_drive(file_id: str, dest_path: str, chunk_size: int = 32768):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    resp = session.get(URL, params={'id': file_id}, stream=True, timeout=60)
    token = _get_confirm_token(resp)
    if token:
        resp = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download file from Drive (status {resp.status_code})")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

# ---------------- Load dataset ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ---------------- Ensure model file exists locally (download if missing) ----------------
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    try:
        print(f"Model not found at {MODEL_PATH}. Attempting to download from Google Drive...")
        download_from_drive(DRIVE_FILE_ID, MODEL_PATH)
        print(f"Downloaded model to {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: could not download model from Drive: {e}. Will continue and build TF-IDF locally if needed.")

# ---------------- Load artifacts if present ----------------
artifacts = {}
if os.path.exists(MODEL_PATH):
    try:
        artifacts = joblib.load(MODEL_PATH) or {}
        print(f"Loaded artifacts from {MODEL_PATH}: keys={list(artifacts.keys())}")
    except Exception as e:
        print(f"Warning: failed to load artifacts from {MODEL_PATH}: {e}")
        artifacts = {}

# ---------------- Reuse vectorizer/tfidf if in artifacts ----------------
vectorizer = artifacts.get('vectorizer')
tfidf_matrix = artifacts.get('tfidf_matrix')

# ---------------- Clean text (for fitting TF-IDF if needed) ----------------
df['clean_text'] = df['body_text'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

# ---------------- Fit TF-IDF if not available ----------------
if vectorizer is None or tfidf_matrix is None:
    print("Fitting TF-IDF vectorizer on extracted_content.csv...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    print("TF-IDF fitting complete.")
else:
    print("Using TF-IDF artifacts loaded from model file.")

# ---------------- Optional: compute top keywords for each doc ----------------
try:
    feature_names = vectorizer.get_feature_names_out()
    def get_top_k_keywords(row_tfidf, k=5):
        if row_tfidf.nnz == 0:
            return ''
        row = row_tfidf.toarray().flatten()
        top_idx = np.argsort(row)[-k:][::-1]
        top_words = [feature_names[i] for i in top_idx if row[i] > 0]
        return '|'.join(top_words)
    df['top_keywords'] = [get_top_k_keywords(tfidf_matrix[i], k=5) for i in range(tfidf_matrix.shape[0])]
except Exception as e:
    print(f"Warning: could not compute top_keywords: {e}")

# ---------------- Feature columns for model ----------------
feature_cols = artifacts.get('feature_cols', ['word_count', 'sentence_count', 'flesch_reading_ease'])

# ---------------- Update & persist artifacts into MODEL_PATH ----------------
artifacts.update({
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'feature_cols': feature_cols
})
try:
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Saved/updated artifacts to {MODEL_PATH} (keys={list(artifacts.keys())})")
except Exception as e:
    print(f"Warning: failed to save artifacts to {MODEL_PATH}: {e}")

# ---------------- Exports ----------------
__all__ = ['df', 'vectorizer', 'tfidf_matrix', 'feature_cols']

# ------------------------------
# Force build / test run
# ------------------------------
if __name__ == "__main__":
    print("Running features.py standalone to force download/build...")
    print(f"Dataset rows: {df.shape[0]}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Feature columns: {feature_cols}")

# streamlit_app/utils/features.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import requests
from io import BytesIO

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # streamlit_app/utils
DATA_PATH = os.path.join(BASE_DIR, '../../data/extracted_content.csv')  # project_root/data
MODEL_DIR = os.path.join(BASE_DIR, '../models')               # streamlit_app/models
MODEL_PATH = os.path.join(MODEL_DIR, 'quality_model.pkl')

# ---------------- Google Drive ----------------
DRIVE_FILE_ID = "1YF7BhTtTZeRab_oltPMANK4dBLhgzJ8C"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

def download_model_from_drive(target_path):
    """Download a file from Google Drive."""
    print(f"Downloading model from Google Drive to {target_path} ...")
    response = requests.get(DRIVE_URL)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

# ---------------- Ensure model exists ----------------
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    download_model_from_drive(MODEL_PATH)

# ---------------- Load dataset ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ---------------- Clean text ----------------
df['clean_text'] = df['body_text'].fillna('').astype(str)\
    .str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

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

# ---------------- Load existing artifacts ----------------
try:
    artifacts = joblib.load(MODEL_PATH)
    print(f"Loaded model artifacts from {MODEL_PATH}: keys={list(artifacts.keys())}")
except Exception:
    artifacts = {}

# ---------------- Update artifacts ----------------
artifacts.update({
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'feature_cols': feature_cols
})

# ---------------- Save updated artifacts ----------------
joblib.dump(artifacts, MODEL_PATH)
print(f"Saved TF-IDF artifacts to {MODEL_PATH}")

# ------------------------------
# Force build / test run
# ------------------------------
if __name__ == "__main__":
    print("Running features.py standalone to force download/build...")
    print(f"Dataset rows: {df.shape[0]}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Feature columns: {feature_cols}")

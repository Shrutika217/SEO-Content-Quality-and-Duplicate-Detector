# streamlit_app/utils/scorer.py
import os
import joblib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import logging

from .parser import parse_html_largest_block

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # streamlit_app/utils
DATA_PATH = os.path.join(BASE_DIR, '../../data/extracted_content.csv')
MODEL_PATH = os.path.join(BASE_DIR, '../models/quality_model.pkl')   # streamlit_app/models/quality_model.pkl

# ---------------- Load dataset ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ---------------- Load artifacts ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifacts not found: {MODEL_PATH}")
artifacts = joblib.load(MODEL_PATH)

clf = artifacts.get('model')                    # may be None
le = artifacts.get('label_encoder')            # may be None
vectorizer = artifacts.get('vectorizer')       # may be None
tfidf_matrix = artifacts.get('tfidf_matrix')   # may be None
feature_cols = artifacts.get('feature_cols', ['word_count', 'sentence_count', 'flesch_reading_ease'])

# If TF-IDF artifacts missing, build fallback (and persist)
if (vectorizer is None or tfidf_matrix is None):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("TF-IDF artifacts missing. Fitting TF-IDF on extracted content as fallback.")
        df['clean_text'] = df['body_text'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['clean_text'].fillna(''))
        artifacts.update({'vectorizer': vectorizer, 'tfidf_matrix': tfidf_matrix})
        joblib.dump(artifacts, MODEL_PATH)
        logger.info("Saved TF-IDF fallback into model artifacts.")
    except Exception as e:
        logger.warning("Failed to build TF-IDF fallback: %s", e)
        vectorizer = None
        tfidf_matrix = None

# Safe defaults for percentiles if missing
wc_90 = artifacts.get('wc_90', int(df['word_count'].quantile(0.90)) if 'word_count' in df.columns else 1000)
flesch_60 = artifacts.get('flesch_60', float(df['flesch_reading_ease'].quantile(0.60)) if 'flesch_reading_ease' in df.columns else 40.0)

# ---------------- Utilities ----------------
def avg_sentence_length(text):
    sents = [s.strip() for s in text.split('.') if s.strip()]
    if not sents:
        return 0.0
    total_words = sum(len(s.split()) for s in sents)
    return total_words / len(sents)

def count_headings_from_html(html):
    try:
        soup = BeautifulSoup(html, 'lxml')
        count = sum(len(soup.find_all(f'h{i}')) for i in range(1, 7))
        return count
    except Exception:
        return 0

def improved_heuristic_label(word_count, flesch, body_text, html):
    avg_sent = avg_sentence_length(body_text)
    headings = count_headings_from_html(html)

    if word_count >= 1500:
        if flesch >= 45 or (avg_sent < 30 and headings >= 3):
            return 'High'
        return 'Medium'

    if word_count >= 800 and flesch >= 45:
        return 'High'
    if word_count >= 300 and flesch >= 35:
        return 'Medium'

    return 'Low'

# ---------------- Main function ----------------
def analyze_url(url: str, top_k_similar: int = 5, model_conf_threshold: float = 0.65):
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error("Failed to fetch %s : %s", url, e)
        return {'error': f'fetch_failed: {e}', 'url': url}

    title, body_text, word_count = parse_html_largest_block(html)
    clean_text = ' '.join(str(body_text).lower().split())
    readability = textstat.flesch_reading_ease(clean_text) if clean_text else 0.0
    sentence_count = max(0, len([s for s in clean_text.split('.') if s.strip()]))
    is_thin = word_count < 250

    feat_vals = []
    for c in feature_cols:
        if c == 'word_count':
            feat_vals.append(word_count)
        elif c == 'sentence_count':
            feat_vals.append(sentence_count)
        elif c in ('flesch_reading_ease', 'readability'):
            feat_vals.append(readability)
        else:
            feat_vals.append(0.0)
    X = pd.DataFrame([feat_vals], columns=feature_cols)

    model_label = None
    model_conf = None
    if clf is not None:
        try:
            pred_enc = clf.predict(X)[0]
            model_label = le.inverse_transform([pred_enc])[0] if le is not None else str(pred_enc)
            probs = clf.predict_proba(X)[0]
            model_conf = float(probs[int(pred_enc)]) if probs is not None else None
            if model_conf is not None:
                model_conf = float(min(model_conf, 0.999999))
        except Exception as e:
            logger.warning("Model predict failed: %s", e)
            model_label = None
            model_conf = None

    h_label = improved_heuristic_label(word_count, readability, body_text, html)
    avg_sent = avg_sentence_length(body_text)
    heading_count = count_headings_from_html(html)

    final_label = h_label
    if model_label is not None and model_conf is not None:
        if model_conf >= 0.75:
            final_label = model_label
        elif model_conf >= model_conf_threshold and model_label == h_label:
            final_label = model_label
        else:
            final_label = h_label

    similar_docs = []
    if vectorizer is not None and tfidf_matrix is not None and clean_text:
        try:
            v = vectorizer.transform([clean_text])
            sims = cosine_similarity(v, tfidf_matrix).flatten()
            idxs = np.argsort(sims)[-top_k_similar:][::-1]
            for i in idxs:
                similar_docs.append({'url': df.loc[i, 'url'], 'similarity': float(sims[i])})
        except Exception as e:
            logger.warning("Similarity computation failed: %s", e)
            similar_docs = []

    if not any(d.get('url') == url for d in similar_docs):
        similar_docs.insert(0, {'url': url, 'similarity': 1.0})
    else:
        similar_docs = [d for d in similar_docs if d.get('url') != url]
        similar_docs.insert(0, {'url': url, 'similarity': 1.0})

    similar_docs = sorted(similar_docs, key=lambda x: x['similarity'], reverse=True)

    result = {
        'url': url,
        'title': title,
        'word_count': int(word_count),
        'sentence_count': int(sentence_count),
        'readability': round(float(readability), 2),
        'quality_label': final_label,
        'is_thin': bool(is_thin),
        'similar_to': similar_docs,
        '_model_confidence_pct': round(model_conf * 100, 2) if model_conf is not None else None,
        'body_text': body_text,
        '_heuristic_label': h_label,
        '_avg_sentence_length': round(avg_sent, 2),
        '_heading_count': int(heading_count),
        '_model_pred_label': model_label,
        '_model_confidence': round(model_conf, 3) if model_conf is not None else None
    }

    return result

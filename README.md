# SEO Content Quality & Duplicate Detector

A **Streamlit-based web application** to analyze webpages for **content quality**, **readability**, and **duplicate content detection**. This tool uses the **Random Forest** model to classify web content as **Low**, **Medium**, or **High** quality, and also identifies similar pages using TF-IDF and cosine similarity.

---

## Features

- **Content Quality Detection (Random Forest)**  
  Uses a **Random Forest classifier** (calibrated with cross-validation) to classify webpages into **Low**, **Medium**, or **High** quality based on extracted features.

- **Readability Analysis**  
  Computes the **Flesch Reading Ease** score to assess how easy the content is to read.

- **Duplicate & Similar Content Detection**  
  Identifies similar or duplicate pages in the dataset using **TF-IDF vectorization** and **cosine similarity**.

- **Thin Content Warning**  
  Flags pages with low word count (<500 words) or very low readability as potentially thin or low-quality content.

- **Google Drive Model Integration**  
  Loads trained model artifacts from **Google Drive**, enabling deployment without including `.pkl` files in the repository.

---

## Model Training Summary

| Metric | Value |
|--------|-------|
| Loaded rows | 81 rows from `../data/data.csv` |
| Parsed content | `../data/extracted_content.csv` |
| Features | `../data/features.csv` |
| Duplicates | `../data/duplicates.csv` (found 24 pairs at threshold 0.8) |
| Adaptive thresholds | `word_count >= 6570`, `flesch >= 36.81` |
| Label distribution | Low: 43, Medium: 32, High: 6 |
| Cross-validation | `cv=4` due to class sizes `Counter({1: 30, 2: 22, 0: 4})` |

---

## Classification Report

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| High   | 0.50      | 0.50   | 0.50     | 2       |
| Low    | 1.00      | 1.00   | 1.00     | 13      |
| Medium | 0.90      | 0.90   | 0.90     | 10      |
| **Accuracy** | - | - | 0.92 | 25 |
| **Macro Avg** | 0.80 | 0.80 | 0.80 | 25 |
| **Weighted Avg** | 0.92 | 0.92 | 0.92 | 25 |

---

## Confusion Matrix

|          | Pred High | Pred Low | Pred Medium |
|----------|-----------|----------|-------------|
| True High    | 1         | 0        | 1           |
| True Low     | 0         | 13       | 0           |
| True Medium  | 1         | 0        | 9           |

---

## Live Demo

You can access the deployed Streamlit app here:  
[SEO Content Quality & Duplicate Detector](https://seo-content-quality-and-duplicate-detector-rhi9vjk7ytzupmznuwu.streamlit.app/)

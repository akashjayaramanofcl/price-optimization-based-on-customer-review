# Intelligent Sentiment Analysis System for Product Reviews

An end-to-end machine learning system that classifies product reviews into **positive**, **neutral**, and **negative** sentiments using advanced NLP techniques — achieving **95.2% accuracy** and a **0.93 macro F1-score**.

---

## Overview

This project automates sentiment classification of e-commerce product reviews. It handles real-world data challenges including class imbalance, noisy text, and encoding issues, and deploys a trained Random Forest model through an interactive Flask web interface.

---

##  Features

- **NLP Preprocessing Pipeline** — tokenization, stop-word removal, lemmatization via NLTK & spaCy
- **Rich Feature Engineering** — TF-IDF (2,000 features + bigrams), TextBlob polarity/subjectivity scores, review length, word count, and scaled price data
- **Class Imbalance Handling** — SMOTE oversampling to balance positive (~65%), negative (~25%), and neutral (~10%) classes
- **Hyperparameter Tuning** — GridSearchCV across 54 Random Forest parameter combinations (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- **Flask Web App** — CSV upload, real-time predictions, interactive dashboards, and downloadable reports
- **Model Persistence** — joblib serialization for production-ready inference

---

##  Results

| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative  | 0.93      | 0.92   | 0.93     |
| Neutral   | 0.88      | 0.88   | 0.88     |
| Positive  | 0.96      | 0.95   | 0.96     |
| **Macro Avg** | **0.92** | **0.92** | **0.93** |

**Overall Accuracy: 95.2%**

---

##  Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | Pandas, NumPy |
| NLP | NLTK, spaCy, TextBlob |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web Framework | Flask |
| Model Persistence | joblib |

---

##  Getting Started

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install nltk wordcloud textblob spacy imbalanced-learn flask
python -m spacy download en_core_web_sm
```

### 2. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Prepare Your Dataset

Your CSV file should contain the following columns:

| Column | Description |
|--------|-------------|
| `Review` | Raw review text |
| `Rate` | Star rating (1–5) |
| `Price` | Product price |

Sentiment labels are auto-generated from ratings: `1–2 → negative`, `3 → neutral`, `4–5 → positive`.

### 4. Train the Model

Run the notebook or script to preprocess data, train, and save the model:

```bash
python train.py
```

This will output:
- `sentiment_model.joblib`
- `tfidf_vectorizer.joblib`
- `scaler.joblib`
- `label_encoder.joblib`

### 5. Launch the Web App

```bash
python app.py
```

Visit `http://localhost:5000` to upload a CSV and get real-time sentiment predictions.

---

##  Project Structure

```
├── train.py                  # Model training pipeline
├── app.py                    # Flask web application
├── sentiment_model.joblib    # Trained Random Forest model
├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
├── scaler.joblib             # Fitted StandardScaler
├── label_encoder.joblib      # Fitted LabelEncoder
├── templates/
│   └── index.html            # Web interface
└── README.md
```

---

##  Methodology

1. **Data Cleaning** — Drop nulls, remove duplicates, normalize price, apply IQR-based outlier capping
2. **Text Preprocessing** — Lowercase, remove special characters, tokenize, filter stop words, lemmatize
3. **Feature Engineering** — Combine TF-IDF vectors with sentiment scores and statistical features (2,005-dimensional feature space)
4. **SMOTE** — Balance class distribution in training data only
5. **Model Training** — GridSearchCV-optimized Random Forest with 3-fold cross-validation
6. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix, feature importance

---

##  Limitations

- Does not handle sarcasm or irony well
- English-only (no multilingual support)
- Requires retraining for different product domains
- Neutral class shows slightly lower performance (F1: 0.88)

---

##  Future Enhancements

- Transformer-based models (BERT / RoBERTa) for contextual understanding
- Aspect-based sentiment analysis for granular insights
- Multilingual support (Spanish, French, Chinese)
- Real-time streaming with Apache Kafka
- Explainable AI with LIME / SHAP
- Active learning for continuous model improvement

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

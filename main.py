"""
=============================================================================
Analisis Performa TF-IDF N-gram (Unigram, Bigram, Trigram)
untuk Klasifikasi Sarkasme pada Dataset Reddit vs Twitter Indonesia
=============================================================================

Research Questions:
1. Apakah N-gram order yang optimal berbeda untuk short-form (Twitter)
   vs long-form (Reddit) text?
2. Kombinasi N-gram mana yang paling efektif?

Datasets:
- Reddit Indonesia Sarcastic  : w11wo/reddit_indonesia_sarcastic  (14,116 comments)
- Twitter Indonesia Sarcastic : w11wo/twitter_indonesia_sarcastic (2,684 tweets)

Method:
- Feature Extraction : TF-IDF with n-gram variations
- Classifiers        : SVM, Random Forest, Logistic Regression
- Evaluation         : Accuracy, Precision, Recall, F1-Score
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from evaluation import (
    create_results_dataframe,
    generate_all_visualizations,
    print_best_configurations,
)
from preprocessing import preprocess_text

warnings.filterwarnings("ignore")

# ============================================================================
# KONFIGURASI EKSPERIMEN
# ============================================================================

# N-gram configurations yang akan diuji
NGRAM_CONFIGS = {
    "Unigram (1,1)":        (1, 1),
    "Bigram (2,2)":         (2, 2),
    "Trigram (3,3)":        (3, 3),
    "Uni+Bi (1,2)":        (1, 2),
    "Uni+Bi+Tri (1,3)":    (1, 3),
    "Bi+Tri (2,3)":        (2, 3),
}

# Classifier configurations
CLASSIFIERS = {
    "SVM": SVC(kernel="linear", C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, n_jobs=-1,
    ),
}

# Cross-validation settings
N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================================
# FUNGSI LOAD DATA
# ============================================================================

def load_reddit_dataset():
    """Memuat dataset Reddit Indonesia Sarcastic dari HuggingFace."""
    print("\n[INFO] Memuat dataset Reddit Indonesia Sarcastic...")
    dataset = load_dataset("w11wo/reddit_indonesia_sarcastic")

    texts, labels = [], []
    for split in dataset:
        for item in dataset[split]:
            texts.append(item["text"])
            labels.append(item["label"])

    print(f"  Total data Reddit: {len(texts)}")
    print(f"  Distribusi label: {pd.Series(labels).value_counts().to_dict()}")
    return texts, labels


def load_twitter_dataset():
    """Memuat dataset Twitter Indonesia Sarcastic dari HuggingFace."""
    print("\n[INFO] Memuat dataset Twitter Indonesia Sarcastic...")
    dataset = load_dataset("w11wo/twitter_indonesia_sarcastic")

    texts, labels = [], []
    for split in dataset:
        for item in dataset[split]:
            texts.append(item["tweet"])
            labels.append(item["label"])

    print(f"  Total data Twitter: {len(texts)}")
    print(f"  Distribusi label: {pd.Series(labels).value_counts().to_dict()}")
    return texts, labels


# ============================================================================
# FUNGSI PREPROCESSING
# ============================================================================

def preprocess_dataset(texts, dataset_name=""):
    """Menjalankan preprocessing pada seluruh dataset."""
    print(f"\n[INFO] Preprocessing dataset {dataset_name}...")
    start_time = time.time()

    processed = []
    total = len(texts)
    for i, text in enumerate(texts):
        processed.append(preprocess_text(text))
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"  Selesai dalam {elapsed:.1f} detik")

    # Statistik teks setelah preprocessing
    lengths = [len(t.split()) for t in processed]
    print(f"  Rata-rata panjang teks: {np.mean(lengths):.1f} kata")
    print(f"  Median panjang teks   : {np.median(lengths):.1f} kata")
    print(f"  Min/Max panjang teks  : {np.min(lengths)}/{np.max(lengths)} kata")

    return processed


# ============================================================================
# FUNGSI EKSPERIMEN
# ============================================================================

def run_experiment(texts, labels, dataset_name, output_dir="results"):
    """
    Menjalankan seluruh eksperimen TF-IDF N-gram untuk satu dataset.

    Parameters
    ----------
    texts : list of str
        Teks yang sudah dipreprocessing.
    labels : list of int
        Label (0 = non-sarcastic, 1 = sarcastic).
    dataset_name : str
        Nama dataset (Reddit / Twitter).
    output_dir : str
        Folder output untuk menyimpan hasil.

    Returns
    -------
    list of dict
        Daftar hasil evaluasi dari semua konfigurasi.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    labels_array = np.array(labels)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    total_experiments = len(NGRAM_CONFIGS) * len(CLASSIFIERS)
    current = 0

    print(f"\n{'#'*70}")
    print(f"  EKSPERIMEN: {dataset_name}")
    print(f"  Total konfigurasi: {total_experiments}")
    print(f"  Cross-validation : {N_FOLDS}-fold Stratified")
    print(f"{'#'*70}")

    for ngram_name, ngram_range in NGRAM_CONFIGS.items():
        print(f"\n{'─'*50}")
        print(f"  N-gram: {ngram_name} -> range={ngram_range}")
        print(f"{'─'*50}")

        for clf_name, clf in CLASSIFIERS.items():
            current += 1
            print(f"\n  [{current}/{total_experiments}] {clf_name} + {ngram_name}")

            start_time = time.time()

            # Pipeline: TF-IDF di-fit hanya pada training fold (no data leakage)
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=50000,
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.95,
                )),
                ("clf", clf),
            ])

            # Cross-validation
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
            cv_results = cross_validate(
                pipeline, texts, labels_array, cv=cv, scoring=scoring,
                return_train_score=False, n_jobs=-1,
            )

            elapsed = time.time() - start_time

            # Rata-rata metrik dari CV
            metric_dict = {
                "dataset": dataset_name,
                "model": clf_name,
                "ngram": ngram_name,
                "accuracy": np.mean(cv_results["test_accuracy"]),
                "precision": np.mean(cv_results["test_precision_weighted"]),
                "recall": np.mean(cv_results["test_recall_weighted"]),
                "f1_score": np.mean(cv_results["test_f1_weighted"]),
                "accuracy_std": np.std(cv_results["test_accuracy"]),
                "f1_std": np.std(cv_results["test_f1_weighted"]),
                "time_seconds": elapsed,
            }
            results.append(metric_dict)

            print(f"    Accuracy : {metric_dict['accuracy']:.4f} "
                  f"(+/- {metric_dict['accuracy_std']:.4f})")
            print(f"    F1-Score : {metric_dict['f1_score']:.4f} "
                  f"(+/- {metric_dict['f1_std']:.4f})")
            print(f"    Precision: {metric_dict['precision']:.4f}")
            print(f"    Recall   : {metric_dict['recall']:.4f}")
            print(f"    Waktu    : {elapsed:.1f}s")

    return results


# ============================================================================
# FUNGSI ANALISIS KOMPARATIF
# ============================================================================

def comparative_analysis(results_df):
    """
    Melakukan analisis komparatif antara dataset Reddit dan Twitter.
    """
    print("\n" + "=" * 70)
    print("  ANALISIS KOMPARATIF: REDDIT vs TWITTER")
    print("=" * 70)

    # 1. N-gram optimal per dataset
    print("\n--- 1. N-gram Optimal per Dataset ---")
    for dataset in results_df["dataset"].unique():
        subset = results_df[results_df["dataset"] == dataset]
        best = subset.loc[subset["f1_score"].idxmax()]
        print(f"\n  {dataset}:")
        print(f"    Best Config : {best['model']} + {best['ngram']}")
        print(f"    F1-Score    : {best['f1_score']:.4f}")

    # 2. Rata-rata performa per N-gram config
    print("\n--- 2. Rata-rata F1-Score per N-gram (across all models) ---")
    ngram_avg = results_df.groupby(["dataset", "ngram"])["f1_score"].mean()
    print(ngram_avg.unstack(level=0).round(4).to_string())

    # 3. Rata-rata performa per model
    print("\n--- 3. Rata-rata F1-Score per Model (across all N-grams) ---")
    model_avg = results_df.groupby(["dataset", "model"])["f1_score"].mean()
    print(model_avg.unstack(level=0).round(4).to_string())

    # 4. Perbedaan short-form vs long-form
    print("\n--- 4. Analisis Short-form (Twitter) vs Long-form (Reddit) ---")
    for ngram in results_df["ngram"].unique():
        reddit_f1 = results_df[
            (results_df["dataset"] == "Reddit") & (results_df["ngram"] == ngram)
        ]["f1_score"].mean()
        twitter_f1 = results_df[
            (results_df["dataset"] == "Twitter") & (results_df["ngram"] == ngram)
        ]["f1_score"].mean()
        diff = reddit_f1 - twitter_f1
        better = "Reddit" if diff > 0 else "Twitter"
        print(f"  {ngram:20s} -> Reddit={reddit_f1:.4f}, "
              f"Twitter={twitter_f1:.4f}, Diff={diff:+.4f} ({better} lebih baik)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Fungsi utama yang menjalankan seluruh pipeline eksperimen."""
    print("=" * 70)
    print("  TF-IDF N-GRAM SARCASM CLASSIFICATION")
    print("  Reddit vs Twitter Indonesia")
    print("=" * 70)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # --- LOAD DAN PREPROCESS DATASET ---
    # Reddit
    reddit_texts, reddit_labels = load_reddit_dataset()
    reddit_processed = preprocess_dataset(reddit_texts, "Reddit")

    # Twitter
    twitter_texts, twitter_labels = load_twitter_dataset()
    twitter_processed = preprocess_dataset(twitter_texts, "Twitter")

    # --- JALANKAN EKSPERIMEN ---
    # Reddit experiments
    reddit_results = run_experiment(
        reddit_processed, reddit_labels, "Reddit", output_dir,
    )
    all_results.extend(reddit_results)

    # Twitter experiments
    twitter_results = run_experiment(
        twitter_processed, twitter_labels, "Twitter", output_dir,
    )
    all_results.extend(twitter_results)

    # --- ANALISIS HASIL ---
    results_df = create_results_dataframe(all_results)

    # Simpan hasil ke CSV
    csv_path = os.path.join(output_dir, "all_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Hasil disimpan ke: {csv_path}")

    # Print tabel hasil lengkap
    print("\n" + "=" * 70)
    print("  TABEL HASIL LENGKAP")
    print("=" * 70)
    display_cols = ["dataset", "model", "ngram", "accuracy", "precision",
                    "recall", "f1_score", "accuracy_std", "f1_std"]
    print(results_df[display_cols].to_string(index=False))

    # Konfigurasi terbaik
    print_best_configurations(results_df)

    # Analisis komparatif
    comparative_analysis(results_df)

    # --- VISUALISASI ---
    print("\n[INFO] Membuat visualisasi...")
    generate_all_visualizations(results_df, output_dir)

    print("\n" + "=" * 70)
    print("  EKSPERIMEN SELESAI!")
    print(f"  Semua hasil tersimpan di folder: {output_dir}/")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = main()

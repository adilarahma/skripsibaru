"""
Modul Evaluasi dan Visualisasi
Untuk analisis performa TF-IDF N-gram pada klasifikasi sarkasme.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true, y_pred, model_name="", ngram_name="", dataset_name=""):
    """
    Mengevaluasi prediksi model dan mengembalikan metrik performa.

    Returns
    -------
    dict
        Dictionary berisi metrik evaluasi.
    """
    metrics = {
        "dataset": dataset_name,
        "model": model_name,
        "ngram": ngram_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def print_classification_report(y_true, y_pred, title=""):
    """Mencetak classification report dengan format rapi."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["Non-Sarcastic", "Sarcastic"]))


def create_results_dataframe(all_results):
    """
    Membuat DataFrame dari semua hasil eksperimen.

    Parameters
    ----------
    all_results : list of dict
        List berisi dictionary metrik dari setiap eksperimen.

    Returns
    -------
    pd.DataFrame
        DataFrame terstruktur berisi semua hasil.
    """
    df = pd.DataFrame(all_results)
    df = df.round(4)
    return df


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Membuat plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Sarcastic", "Sarcastic"],
        yticklabels=["Non-Sarcastic", "Sarcastic"],
    )
    plt.title(title, fontsize=13)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ngram_comparison(results_df, metric="f1_score", save_path=None):
    """
    Membuat bar chart perbandingan performa N-gram per model dan dataset.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame hasil evaluasi.
    metric : str
        Metrik yang akan divisualisasikan.
    save_path : str or None
        Path untuk menyimpan gambar.
    """
    datasets = results_df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6), sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = results_df[results_df["dataset"] == dataset]
        pivot = subset.pivot(index="ngram", columns="model", values=metric)

        pivot.plot(kind="bar", ax=ax, rot=45, colormap="Set2", edgecolor="black")
        ax.set_title(f"{dataset} - {metric.upper()} per N-gram", fontsize=13)
        ax.set_xlabel("N-gram Configuration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(title="Model", fontsize=9)
        ax.set_ylim(0, 1.05)

        # Tambahkan nilai di atas bar
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=3)

    plt.suptitle(
        f"Perbandingan Performa {metric.upper()} - TF-IDF N-gram Klasifikasi Sarkasme",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_comparison(results_df, metric="f1_score", save_path=None):
    """
    Membuat grouped bar chart perbandingan antar dataset.
    """
    models = results_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), sharey=True)

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = results_df[results_df["model"] == model]
        pivot = subset.pivot(index="ngram", columns="dataset", values=metric)

        pivot.plot(kind="bar", ax=ax, rot=45, colormap="Set1", edgecolor="black")
        ax.set_title(f"{model} - {metric.upper()} Reddit vs Twitter", fontsize=13)
        ax.set_xlabel("N-gram Configuration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(title="Dataset", fontsize=9)
        ax.set_ylim(0, 1.05)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=3)

    plt.suptitle(
        f"Reddit vs Twitter - {metric.upper()} Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap_summary(results_df, metric="f1_score", save_path=None):
    """
    Membuat heatmap ringkasan seluruh eksperimen.
    """
    datasets = results_df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5))

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = results_df[results_df["dataset"] == dataset]
        pivot = subset.pivot(index="model", columns="ngram", values=metric)

        sns.heatmap(
            pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, vmin=0.5, vmax=1.0,
        )
        ax.set_title(f"{dataset} - {metric.upper()} Heatmap", fontsize=13)
        ax.set_xlabel("N-gram Configuration")
        ax.set_ylabel("Model")

    plt.suptitle(
        f"Heatmap {metric.upper()} - Semua Eksperimen",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(results_df, output_dir="results"):
    """
    Menghasilkan semua visualisasi dan menyimpannya ke folder output.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame berisi semua hasil evaluasi.
    output_dir : str
        Folder output untuk menyimpan gambar.
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric in ["accuracy", "f1_score", "precision", "recall"]:
        plot_ngram_comparison(
            results_df, metric=metric,
            save_path=os.path.join(output_dir, f"ngram_comparison_{metric}.png"),
        )
        plot_dataset_comparison(
            results_df, metric=metric,
            save_path=os.path.join(output_dir, f"dataset_comparison_{metric}.png"),
        )
        plot_heatmap_summary(
            results_df, metric=metric,
            save_path=os.path.join(output_dir, f"heatmap_{metric}.png"),
        )

    print(f"\nSemua visualisasi disimpan di folder: {output_dir}/")


def print_best_configurations(results_df):
    """Mencetak konfigurasi terbaik per dataset."""
    print("\n" + "=" * 70)
    print("  KONFIGURASI TERBAIK PER DATASET (berdasarkan F1-Score)")
    print("=" * 70)

    for dataset in results_df["dataset"].unique():
        subset = results_df[results_df["dataset"] == dataset]
        best_row = subset.loc[subset["f1_score"].idxmax()]
        print(f"\n  Dataset: {dataset}")
        print(f"  Model terbaik   : {best_row['model']}")
        print(f"  N-gram terbaik  : {best_row['ngram']}")
        print(f"  Accuracy        : {best_row['accuracy']:.4f}")
        print(f"  F1-Score        : {best_row['f1_score']:.4f}")
        print(f"  Precision       : {best_row['precision']:.4f}")
        print(f"  Recall          : {best_row['recall']:.4f}")

    print("\n" + "=" * 70)
    print("  PERBANDINGAN N-GRAM OPTIMAL: SHORT-FORM vs LONG-FORM")
    print("=" * 70)

    for model in results_df["model"].unique():
        print(f"\n  Model: {model}")
        for dataset in results_df["dataset"].unique():
            subset = results_df[
                (results_df["dataset"] == dataset) & (results_df["model"] == model)
            ]
            best = subset.loc[subset["f1_score"].idxmax()]
            print(f"    {dataset:10s} -> N-gram optimal: {best['ngram']:15s} "
                  f"(F1={best['f1_score']:.4f})")

    print()

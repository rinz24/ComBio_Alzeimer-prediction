import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_single_model_graphs(acc, f1, auc, cm, model_name):

    # ===========================
    # ACCURACY BAR
    plt.figure(figsize=(6,4))
    plt.bar(["Accuracy"], [acc], color="pink", edgecolor="black")
    plt.ylim(0, 1)
    plt.title(f"Accuracy – {model_name}")
    plt.text(0, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

    # ===========================
    # F1 SCORE BAR
    plt.figure(figsize=(6,4))
    plt.bar(["F1 Score"], [f1], color="yellow", edgecolor="black")
    plt.ylim(0, 1)
    plt.title(f"F1 Score – {model_name}")
    plt.text(0, f1 + 0.02, f"{f1:.3f}", ha="center", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

    # ===========================
    # AUC BAR
    plt.figure(figsize=(6,4))
    plt.bar(["AUC"], [auc], color="#FF69B4", edgecolor="black")
    plt.ylim(0, 1)
    plt.title(f"AUC – {model_name}")
    plt.text(0, auc + 0.02, f"{auc:.3f}", ha="center", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.show()

    # ===========================
    # CONFUSION MATRIX HEATMAP
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r",
                linewidths=1, linecolor="black", cbar=False)
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

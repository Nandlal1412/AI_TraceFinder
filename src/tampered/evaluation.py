# src/evaluation.py
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np

def compute_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)

def plot_roc(y_true, y_prob, save=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    if save: plt.savefig(save)
    plt.show()

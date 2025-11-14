# src/training.py
import os, pickle, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

def fit_patch_svm(X, y, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    base = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced")
    clf = CalibratedClassifierCV(base, cv=5, method="sigmoid").fit(Xs, y)
    with open(os.path.join(model_dir, "patch_scaler.pkl"), "wb") as f: pickle.dump(sc, f)
    with open(os.path.join(model_dir, "patch_svm_sig_calibrated.pkl"), "wb") as f: pickle.dump(clf, f)
    probs = clf.predict_proba(Xs)[:,1]
    auc = roc_auc_score(y, probs)
    print("Patch SVM AUC:", auc)
    return sc, clf

def compute_threshold(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])

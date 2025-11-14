# src/testing.py
import os, pickle, numpy as np
from .utils import load_to_residual, extract_patches, make_feat_vector

def load_patch_model(model_dir):
    with open(os.path.join(model_dir, "patch_scaler.pkl"), "rb") as f: sc = pickle.load(f)
    with open(os.path.join(model_dir, "patch_svm_sig_calibrated.pkl"), "rb") as f: clf = pickle.load(f)
    return sc, clf

def predict_image_tamper(path, model_dir, thr=0.5, topk_frac=0.3, max_patches=16):
    sc, clf = load_patch_model(model_dir)
    res = load_to_residual(path)
    patches, coords = extract_patches(res, limit=max_patches)
    feats = [make_feat_vector(p) for p in patches]
    if not feats: return {"tamper_score": 0.0, "is_tampered": False}
    X = np.asarray(feats, np.float32)
    Xs = sc.transform(X)
    probs = clf.predict_proba(Xs)[:,1]
    k = max(1, int(len(probs)*topk_frac))
    score = float(np.mean(np.sort(probs)[-k:]))
    return {"tamper_score": score, "is_tampered": bool(score >= thr), "probs": probs, "coords": coords}

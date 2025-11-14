# app.py — Colab-aligned; scanner 27-D; image-level tamper 18-D via patch-average; patch fallback 22-D
import streamlit as st
import os, json, math, pickle, tempfile
import numpy as np
import cv2, pywt, tensorflow as tf
from PIL import Image
from skimage.feature import local_binary_pattern


# --------------------------
# Artifact paths (corrected for: Codes/models/tampered module)
# --------------------------
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "tampered module"
)


# Scanner-ID (27-D handcrafted)
SCN_MODEL_PATH  = os.path.join(BASE_DIR, "scanner_hybrid.keras")
SCN_LE_PATH     = os.path.join(BASE_DIR, "hybrid_label_encoder.pkl")
SCN_SCALER_PATH = os.path.join(BASE_DIR, "hybrid_feat_scaler.pkl")     # expects 27
SCN_FP_PATH     = os.path.join(BASE_DIR, "scanner_fingerprints.pkl")
SCN_FP_KEYS     = os.path.join(BASE_DIR, "fp_keys.npy")                # length 11

# Image-level tamper (18-D via patch-average)
IMG_SCALER_PATH = os.path.join(BASE_DIR, "image_scaler.pkl")           # expects 18
IMG_CLF_PATH    = os.path.join(BASE_DIR, "image_svm_sig.pkl")
IMG_THR_JSON    = os.path.join(BASE_DIR, "image_thresholds.json")      # global/by_domain

# Patch-level tamper fallback (22-D per patch)
TP_SCALER_PATH  = os.path.join(BASE_DIR, "patch_scaler.pkl")
TP_CLF_PATH     = os.path.join(BASE_DIR, "patch_svm_sig_calibrated.pkl")
TP_THR_JSON     = os.path.join(BASE_DIR, "thresholds_patch.json")

# --------------------------
# Constants (match notebook)
# --------------------------
IMG_SIZE    = (256, 256)
PATCH       = 128
STRIDE      = 64
MAX_PATCHES = 16

# --------------------------
# Cached loaders
# --------------------------
@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_numpy_list(path):
    return np.load(path, allow_pickle=True).tolist()

@st.cache_resource
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# --------------------------
# Load artifacts
# --------------------------
# Scanner
hyb_model   = load_tf_model(SCN_MODEL_PATH)
le_sc       = load_pickle(SCN_LE_PATH)
sc_sc       = load_pickle(SCN_SCALER_PATH)
scanner_fps = load_pickle(SCN_FP_PATH)
fp_keys     = load_numpy_list(SCN_FP_KEYS)

# Image-level tamper
HAS_IMG = True
try:
    sc_img = load_pickle(IMG_SCALER_PATH)
    img_clf = load_pickle(IMG_CLF_PATH)
    THR_IMG = load_json(IMG_THR_JSON)
except Exception:
    HAS_IMG = False
    sc_img = None; img_clf = None; THR_IMG = None

# Patch fallback
HAS_PATCH = True
try:
    sc_tp = load_pickle(TP_SCALER_PATH)
    clf_tp = load_pickle(TP_CLF_PATH)
    THR_TP = load_json(TP_THR_JSON)
except Exception:
    HAS_PATCH = False
    sc_tp = None; clf_tp = None; THR_TP = None

# --------------------------
# Residual preprocessing
# --------------------------
def preprocess_residual_pywt(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (img - den).astype(np.float32)

# --------------------------
# Feature utilities
# --------------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins = np.linspace(0, r.max() + 1e-6, K + 1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32).tolist()

# --------------------------
# Scanner 27-D handcrafted
# --------------------------
def make_scanner_feats_from_res(res):
    if len(fp_keys) != 11:
        raise RuntimeError(f"fp_keys length {len(fp_keys)} != 11")
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]  # 11
    v_fft  = fft_radial_energy(res, 6)                       # 6
    v_lbp  = lbp_hist_safe(res, 8, 1.0)                      # 10
    feat = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)  # (1,27)
    n_in = int(getattr(sc_sc, "n_features_in_", 0))
    if n_in != 27:
        raise RuntimeError(f"Scanner scaler expects 27, got {n_in}")
    return sc_sc.transform(feat)

def predict_scanner_hybrid(image_path):
    res = preprocess_residual_pywt(image_path)
    x_img = np.expand_dims(res, axis=(0, -1))
    x_ft  = make_scanner_feats_from_res(res)
    prob  = hyb_model.predict([x_img, x_ft], verbose=0).ravel()
    idx   = int(np.argmax(prob))
    label = le_sc.classes_[idx]
    conf  = float(prob[idx] * 100.0)
    return label, conf

# --------------------------
# Image-level tamper (18-D via patch-average)
#   per patch: 10 LBP + 6 FFT + 2 contrast stats [std, mean(|x| - mean)]
#   image vector = mean over MAX_PATCHES
# --------------------------
def contrast_stat_single(img):
    return np.asarray([float(img.std()),
                       float(np.mean(np.abs(img) - np.mean(img)))], dtype=np.float32)  # 2

def image_patch_feat_18(patch):
    lbp10 = np.asarray(lbp_hist_safe(patch, 8, 1.0), np.float32)  # 10
    fft6  = np.asarray(fft_radial_energy(patch, 6), np.float32)   # 6
    cs2   = contrast_stat_single(patch)                           # 2
    return np.concatenate([lbp10, fft6, cs2], 0)                  # 18

def extract_patches(res, patch=PATCH, stride=STRIDE, limit=MAX_PATCHES, seed=1234):
    H, W = res.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    coords = [(y, x) for y in ys for x in xs]
    rng = np.random.RandomState(seed)
    rng.shuffle(coords)
    coords = coords[:min(limit, len(coords))]
    return [res[y:y+patch, x:x+patch] for (y, x) in coords]

def make_image_feat_18_from_residual(residual_img, seed=1234):
    patches = extract_patches(residual_img, limit=MAX_PATCHES, seed=seed)
    if not patches:
        feat = np.zeros((1, 18), dtype=np.float32)
    else:
        feats = np.stack([image_patch_feat_18(p) for p in patches], 0)   # (N,18)
        feat  = feats.mean(axis=0, keepdims=True).astype(np.float32)     # (1,18)
    n_in = int(getattr(sc_img, "n_features_in_", 0)) if sc_img is not None else 0
    if n_in and n_in != feat.shape[1]:
        raise RuntimeError(f"Image scaler expects {n_in}, built {feat.shape[1]}")
    return feat

def choose_thr_image(domain):
    if THR_IMG is None:
        return 0.5
    return THR_IMG.get("by_domain", {}).get(domain, THR_IMG.get("global", 0.5))

def infer_tamper_image(image_path):
    if not HAS_IMG:
        raise RuntimeError("Image-level artifacts missing.")
    res = preprocess_residual_pywt(image_path)
    x   = make_image_feat_18_from_residual(res, seed=1234)
    x   = sc_img.transform(x)
    p   = float(img_clf.predict_proba(x)[:, 1][0])

    # Domain detection (match Colab rule)
    if ("Originals_tif" in image_path) or ("/Originals_tif/" in image_path):
        dom = "orig_pdf_tif"
    elif ("TamperedImages" in image_path) or ("/TamperedImages/" in image_path):
        dom = "tamper_dir"
    else:
        dom = "tamper_dir"

    thr_raw = choose_thr_image(dom)
    # IMPORTANT: cap per-domain threshold to global for image-level
    thr = min(thr_raw, THR_IMG.get("global", thr_raw))

    tampered = int(p >= thr)
    conf = float((p if tampered else 1.0 - p) * 100.0)

    # Debug
    st.sidebar.write({
        "img_prob": round(p, 3),
        "img_thr_raw": round(thr_raw, 3),
        "img_thr_used": round(thr, 3),
        "domain": dom,
        "img_scaler_n_in": int(getattr(sc_img, "n_features_in_", -1))
    })

    return {
        "prob_tampered": p,
        "tamper_label": "Tampered" if tampered else "Clean",
        "threshold": thr,
        "domain": dom,
        "confidence": conf,
        "hits": -1
    }

# --------------------------
# Patch fallback (22-D per patch, top-k aggregation)
# --------------------------
def residual_stats(img):
    return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    b1 = (r >= 0.25 * rmax) & (r < 0.35 * rmax)
    b2 = (r >= 0.35 * rmax) & (r < 0.50 * rmax)
    e1 = float(mag[b1].mean() if b1.any() else 0.0)
    e2 = float(mag[b2].mean() if b2.any() else 0.0)
    ratio = float(e2 / (e1 + 1e-8))
    return np.asarray([e1, e2, ratio], dtype=np.float32)

def make_patch_feat_22(img_patch):
    lbp = np.asarray(lbp_hist_safe(img_patch, 8, 1.0), np.float32)  # 10
    fft6 = np.asarray(fft_radial_energy(img_patch, 6), np.float32)  # 6
    res3 = residual_stats(img_patch)                                # 3
    rsp3 = fft_resample_feats(img_patch)                            # 3
    return np.concatenate([lbp, fft6, res3, rsp3], 0)               # 22

def image_score_topk(patch_probs, frac=0.30):
    n = len(patch_probs); k = max(1, int(math.ceil(frac * n)))
    top = np.sort(np.asarray(patch_probs))[-k:]
    return float(np.mean(top))

def choose_thr_patch(domain):
    if THR_TP is None:
        return 0.5
    return THR_TP.get("by_domain", {}).get(domain, THR_TP.get("global", 0.5))

def infer_tamper_single_patch(image_path, frac=0.30, local_gate=0.85, min_hits=2, seed=123):
    if not HAS_PATCH:
        raise RuntimeError("Patch-level artifacts missing.")
    res = preprocess_residual_pywt(image_path)
    patches = extract_patches(res, limit=MAX_PATCHES, seed=seed)
    feats = np.stack([make_patch_feat_22(p) for p in patches], 0)
    feats = sc_tp.transform(feats)
    p_patch = clf_tp.predict_proba(feats)[:, 1]
    p_img = image_score_topk(p_patch, frac=frac)
    dom = "orig_pdf_tif" if "/Originals_tif/" in image_path else "tamper_dir"
    thr = choose_thr_patch(dom)
    hits = int((p_patch >= local_gate).sum())
    tampered = int((p_img >= thr) and (hits >= min_hits))
    conf = float((p_img if tampered else 1.0 - p_img) * 100.0)
    st.sidebar.write({"patch_prob_img": round(p_img, 3), "patch_thr": round(thr, 3), "hits": hits, "domain": dom})
    return {
        "prob_tampered": p_img,
        "tamper_label": "Tampered" if tampered else "Clean",
        "threshold": thr,
        "domain": dom,
        "confidence": conf,
        "hits": hits
    }

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Scanner & Tamper (Colab-aligned)", layout="centered")
st.title(" AI-TraceFinder")

uploaded = st.file_uploader("Upload an image (TIFF/PNG/JPG)", type=["png", "jpg", "jpeg", "tif", "tiff"])
run = st.button("Run", type="primary", disabled=(uploaded is None))

if run and uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read()); tmp.close()
    try:
        # Scanner (27-D)
        s_label, s_conf = predict_scanner_hybrid(tmp.name)
        # Tamper: image-level preferred, else patch fallback
        if HAS_IMG:
            t_res = infer_tamper_image(tmp.name)
        elif HAS_PATCH:
            t_res = infer_tamper_single_patch(tmp.name)
        else:
            raise RuntimeError("No tamper artifacts found (image or patch).")

        img = Image.open(tmp.name).convert("RGB")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption=f"Scanner: {s_label} ({s_conf:.2f}%)", use_container_width=True)
        with c2:
            st.metric("Tamper label", t_res["tamper_label"])
            st.write(f"Probability: {t_res['prob_tampered']:.3f}")
            st.write(f"Threshold: {t_res['threshold']:.3f}")
            if "hits" in t_res:
                st.write(f"Hits: {t_res['hits']}")
            st.write(f"Confidence: {t_res['confidence']:.1f}%")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

st.caption(
    "Scanner: 27-D (11 corr + 6 FFT + 10 LBP). Image tamper: 18-D per-patch avg with [std, mean(|x|−mean)], "
    "threshold uses min(domain, global). Patch fallback: 22-D per patch."
)

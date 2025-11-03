# ==============================================================
# FORGERY DETECTION — UNIFIED STREAMLIT APP (UI-ENHANCED)
# Combines Baseline ML + CNN Model + Fingerprint Correlation
# Original functionality preserved; UI/UX improved.
# ==============================================================

# ---------- IMPORTS ----------
import os
import uuid
import pickle
import warnings
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis, entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import pywt

# ---------- Page config ----------
st.set_page_config(page_title="Forgery Detection — Merged App", layout="wide")

# ---------- Globals/Paths ----------
ROOT = Path(".")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)

CSV_PATH = str(ROOT / "official.csv")  # baseline feature CSV (kept as in both apps)
RESIDUALS_PKL = str(ROOT / "Residuals_Paths" / "official_wiki_residuals.pkl")
FLATFIELD_PKL = str(ROOT / "Residuals_Paths" / "flatfield_residuals.pkl")
FP_PKL = str(ROOT / "Residuals_Paths" / "scanner_fingerprints.pkl")
FP_KEYS_NPY = str(ROOT / "Residuals_Paths" / "fp_keys.npy")
CNN_MODEL_DEFAULT = "dual_branch_cnn.h5"

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMG_SIZE = (256, 256)

# TensorFlow is optional & imported lazily
tf = None

# ---------- UI / THEME (CSS) ----------
BASE_CSS = """
<style>
/* Page background and font */
.reporting-app {
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar tweaks */
[data-testid="stSidebar"] .css-1d391kg {
  padding-top: 1rem;
}
.stSidebar .sidebar-content {
  background: linear-gradient(180deg,#0f172a 0%, #0b1220 100%);
  color: white;
}

/* Card container */
.card {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: 0 4px 12px rgba(2,6,23,0.35);
  border: 1px solid rgba(255,255,255,0.04);
}

/* Colored small headers */
.section-header {
  font-size: 18px;
  font-weight: 700;
  color: #0f172a;
}

/* Subtle metric style */
.metric-box {
  padding: 10px;
  border-radius: 8px;
  color: white;
  text-align: center;
}

/* Make dataframe fit nicely */
.stDataFrameContainer {
  border-radius: 8px;
  overflow: hidden;
}

/* Buttons */
.stButton>button {
  border-radius: 8px;
}

/* Small caption style */
.small-caption {
  color: #6b7280;
  font-size: 12px;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# helper for colored metric boxes
def colored_metric(label: str, value: str, bg_color: str = "#4f46e5"):
    st.markdown(
        f"""<div class="metric-box" style="background:{bg_color}">
            <div style="font-size:12px;opacity:0.85">{label}</div>
            <div style="font-size:22px;font-weight:700;margin-top:6px">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------- Small utilities ----------
def safe_load_pickle(path: str):
    try:
        if path and Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None

def safe_load_npy(path: str):
    try:
        if path and Path(path).exists():
            return np.load(path, allow_pickle=True)
    except Exception:
        return None
    return None

def load_tf_model(model_path: str):
    global tf
    try:
        if tf is None:
            import tensorflow as tf_mod
            globals()["tf"] = tf_mod
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return None

def residual_to_display(res: np.ndarray) -> np.ndarray:
    """Pleasant residual rendering (CLAHE + invert) — from both apps, consolidated."""
    a = np.nan_to_num(res.astype(np.float32))
    a = np.abs(a)
    if a.size == 0:
        return a.astype(np.uint8)
    vmin, vmax = np.percentile(a, (1, 99))
    a = np.clip(a, vmin, vmax)
    rng = max(vmax - vmin, 1e-9)
    a = (a - vmin) / rng
    a = np.power(a, 0.5)
    a_uint8 = (a * 255).astype(np.uint8)
    try:
        a_white = cv2.bitwise_not(a_uint8)
    except Exception:
        a_white = 255 - a_uint8
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(a_white)
    except Exception:
        return a_white

def corr2d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

def preprocess_residual_pywt_from_image_path(path: str) -> np.ndarray:
    """DWT denoise residual construction (haar). Uses TF decode for non-TIFF as in both apps."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".tif", ".tiff"]:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError(f"Unable to read {path} as TIFF")
            arr = cv2.resize(arr, IMG_SIZE)
            arr = arr.astype(np.float32) / 255.0
        else:
            global tf
            if tf is None:
                import tensorflow as tf_mod
                tf = tf_mod
            raw = tf.io.read_file(path)
            img = tf.io.decode_image(raw, channels=1, dtype=tf.float32)
            img = tf.image.resize(img, IMG_SIZE)
            arr = img.numpy().squeeze()
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")

    cA, (cH, cV, cD) = pywt.dwt2(arr, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    res = (arr - den).astype(np.float32)
    return res

# ------------------------ Feature Extraction (baseline handcrafted) ------------------------
def extract_basic_features_from_image(image_path: str, class_label: str = "unknown") -> Dict[str, Any]:
    """Handcrafted baseline features — kept from final_app with safe guards."""
    try:
        pil = Image.open(image_path).convert("L")
        gray = np.array(pil).astype(np.float32)
        h, w = gray.shape
        file_size = os.path.getsize(image_path) / 1024.0
        aspect_ratio = round(w / h, 3) if h != 0 else 0
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        sk = float(skew(gray.flatten()))
        ku = float(kurtosis(gray.flatten()))
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = float(entropy(hist + 1e-9))
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = float(np.mean(edges > 0))
        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": int(w),
            "height": int(h),
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 4),
            "std_intensity": round(std_intensity, 4),
            "skewness": round(sk, 3),
            "kurtosis": round(ku, 3),
            "entropy": round(shannon_entropy, 4),
            "edge_density": round(edge_density, 4),
        }
    except Exception as e:
        return {"file_name": image_path, "class": class_label, "error": str(e)}

# ------------------------ Baseline model helpers ------------------------
def train_baseline_models(csv_path: str = CSV_PATH, models_dir: str = "models"):
    """Train RF & SVM, save models+scaler+eval, return artifacts (feat_importances, OOB)."""
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if "file_name" in df.columns and "class" in df.columns:
        X = df.drop(columns=["file_name", "class"])
        y = df["class"]
    else:
        raise ValueError("CSV must contain 'file_name', 'class', and feature columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train_s, y_train)
    joblib.dump(svm, os.path.join(models_dir, "svm.pkl"))

    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # Save quick eval
    y_pred_rf = rf.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
    eval_art = {
        "classes": list(rf.classes_),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True)
    }
    with open(os.path.join(models_dir, "baseline_eval.pkl"), "wb") as f:
        pickle.dump(eval_art, f)

    feat_imp = None
    if hasattr(rf, "feature_importances_"):
        feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {"rf": rf, "svm": svm, "scaler": scaler, "eval": eval_art,
            "feat_imp": feat_imp, "oob": getattr(rf, "oob_score_", None)}

def evaluate_baseline_model(model_path: str, csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]
    scaler = joblib.load(str(MODELS_DIR / "scaler.pkl"))
    model = joblib.load(model_path)
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=getattr(model, "classes_", None) or np.unique(y))
    return report, cm, getattr(model, "classes_", np.unique(y))

def predict_scanner_baseline(img_path: str, model_choice: str = "rf"):
    """Extract baseline features from a single raw image and predict with RF/SVM."""
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.pkl missing in models/")
    scaler = joblib.load(scaler_path)
    model = joblib.load(MODELS_DIR / ("random_forest.pkl" if model_choice == "rf" else "svm.pkl"))

    pil_img = Image.open(img_path).convert("L")
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img.shape
    aspect_ratio = w / h if h != 0 else 0
    file_size_kb = os.path.getsize(img_path) / 1024.0
    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    sk = skew(pixels)
    ku = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)
    edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edge_density = np.mean(edges > 0)

    features = pd.DataFrame([{
        "width": w, "height": h, "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb, "mean_intensity": mean_intensity,
        "std_intensity": std_intensity, "skewness": sk, "kurtosis": ku,
        "entropy": ent, "edge_density": edge_density
    }])

    Xs = scaler.transform(features)
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0] if hasattr(model, "predict_proba") else None
    return pred, prob

# ------------------------ Hybrid helpers (CNN + handcrafted) ------------------------
def make_feats_from_res_for_infer(res: np.ndarray, fingerprints, fp_keys_local) -> np.ndarray:
    """Handcrafted vector built from residual & scanner fingerprints (corr + radial FFT)."""
    try:
        keys = list(fp_keys_local)
    except Exception:
        keys = list(fingerprints.keys())

    v_corr = [corr2d(res, fingerprints[k]) for k in keys]

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K + 1)
        feats = []
        for i in range(K):
            m = (r >= bins[i]) & (r < bins[i + 1])
            feats.append(float(mag[m].mean() if m.any() else 0.0))
        return feats

    v_fft = fft_radial_energy(res, K=6)
    v_lbp = [0.0] * 10  # placeholder LBP slot to match shapes used previously
    return np.array(v_corr + v_fft + v_lbp, dtype=np.float32)

def infer_from_array(model, res_arr: np.ndarray, fingerprints, fp_keys):
    """Unified predictor for 1- or 2-input Keras models; returns (label, conf, raw_probs)."""
    if model is None:
        raise RuntimeError("Model not loaded")

    # image input
    res_rgb = np.stack([res_arr] * 3, axis=-1)  # (256,256,3)
    x_img = np.expand_dims(res_rgb, axis=0).astype(np.float32)

    handcrafted_ready = (fingerprints is not None) and (fp_keys is not None)
    if not handcrafted_ready:
        feats = np.zeros((1, 32), dtype=np.float32)
    else:
        feats_raw = make_feats_from_res_for_infer(res_arr, fingerprints, fp_keys)
        feats = feats_raw.reshape(1, -1).astype(np.float32)

    n_inputs = len(model.inputs)
    try:
        if n_inputs == 1:
            preds = model.predict(x_img, verbose=0)
        elif n_inputs == 2:
            try:
                preds = model.predict([x_img, feats], verbose=0)
            except Exception:
                preds = model.predict([feats, x_img], verbose=0)
        else:
            raise ValueError(f"Unexpected number of model inputs: {n_inputs}")
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs) * 100.0)
    label = str(idx)
    # optional label encoder
    le_path = Path("Residuals_Paths") / "hybrid_label_encoder.pkl"
    if le_path.exists():
        try:
            with open(le_path, "rb") as f:
                le = pickle.load(f)
            if hasattr(le, "classes_") and idx < len(le.classes_):
                label = le.classes_[idx]
        except Exception:
            pass
    return label, conf, probs

# ------------------------ Sidebar (no dropdown — fixed items) ------------------------
st.sidebar.title("Navigation")
SECTION = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Feature Extraction",
        "Dataset Overview",
        "Feature Visualization",
        "Model Training & Evaluation",
        "Live Prediction",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data / Model Paths")
CSV_PATH = st.sidebar.text_input("Baseline CSV path", value=CSV_PATH)
RESIDUALS_PKL = st.sidebar.text_input("Residuals pickle", value=RESIDUALS_PKL)
FP_PKL = st.sidebar.text_input("Fingerprints pickle", value=FP_PKL)
FP_KEYS_NPY = st.sidebar.text_input("fp_keys.npy", value=FP_KEYS_NPY)
CNN_MODEL_DEFAULT = st.sidebar.text_input("Default CNN (.h5)", value=CNN_MODEL_DEFAULT)
st.sidebar.markdown("---")
st.sidebar.caption("Last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ------------------------ Preload pickles (non-blocking) ------------------------
residuals = safe_load_pickle(RESIDUALS_PKL)
fingerprints = safe_load_pickle(FP_PKL)
fp_keys = safe_load_npy(FP_KEYS_NPY)

# ========================================================
#                      UI SECTIONS (ENHANCED)
# ========================================================

def ui_home(residuals_obj, fps, fpks):
    st.title("Home — Quick Overview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    # nice metrics
    with col1:
        bg = "#0ea5a4"  # teal
        total_datasets = len(residuals_obj) if residuals_obj else 0
        colored_metric("Datasets", str(total_datasets), bg)
    with col2:
        bg = "#ef4444"  # red
        total_scanners = sum(len(residuals_obj[ds]) for ds in residuals_obj.keys()) if residuals_obj else 0
        colored_metric("Scanners", str(total_scanners), bg)
    with col3:
        bg = "#4f46e5"  # indigo
        total_images = 0
        if residuals_obj:
            for ds in residuals_obj:
                for sc in residuals_obj[ds]:
                    for dpi in residuals_obj[ds][sc]:
                        total_images += len(residuals_obj[ds][sc][dpi])
        colored_metric("Total residuals", str(total_images), bg)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Dataset Snapshot")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if residuals_obj:
            # show a small table + counts
            dataset_summary = []
            for ds, sc_map in residuals_obj.items():
                total = sum(len(arrs) for sc in sc_map.values() for arrs in sc.values())
                dataset_summary.append({"dataset": ds, "residuals": total, "scanners": len(sc_map)})
            df_summary = pd.DataFrame(dataset_summary).sort_values("residuals", ascending=False)
            # styled table
            styled = (
                    df_summary.style
                    .background_gradient(cmap="Blues", axis=None)  # Softer blue tone
                    .set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#1E3A8A'),
                                                        ('color', 'white'),
                                                        ('font-weight', 'bold')]},
                        {'selector': 'tbody td', 'props': [('border', '1px solid #ddd'),
                                                        ('text-align', 'center')]},
                        {'selector': 'tbody tr:hover', 'props': [('background-color', '#f5f5f5')]}
                    ])
                    .format(precision=0)
                )
            st.dataframe(styled, use_container_width=True)

        else:
            st.info("No residuals loaded. Add Residuals pickle path in the sidebar.")

    with col_b:
        st.markdown("**Quick tips**")
        st.markdown("- Upload the pickles (residuals & fingerprints) in the sidebar paths.")
        st.markdown("- Use *Feature Extraction* to create `official.csv` if missing.")
        st.markdown("- Train baseline models from *Model Training & Evaluation*.")

    st.markdown("---")
    st.subheader("Sample residuals gallery")
    if residuals_obj:
        # gather samples
        samples = []
        for ds, sc_map in residuals_obj.items():
            for sc, dpi_map in sc_map.items():
                for dpi, arrs in dpi_map.items():
                    if arrs:
                        samples.append((ds, residual_to_display(arrs[0])))
                    if len(samples) >= 12:
                        break
                if len(samples) >= 12:
                    break
            if len(samples) >= 12:
                break

        # Display in nice grid cards
        per_row = 4
        for i in range(0, len(samples), per_row):
            cols = st.columns(per_row)
            for j, (caption, img) in enumerate(samples[i:i+per_row]):
                with cols[j]:
                    st.image(img, use_column_width=True, caption=f"{caption}", clamp=True)
    else:
        st.info("No residuals preloaded. Provide pickles in the sidebar.")

def ui_feature_extraction():
    st.title("🧰 Feature Extraction (Baseline)")
    st.markdown("Extract handcrafted features from uploaded images and save to a CSV (default: `official.csv`).")
    st.caption("Inputs: .png/.jpg/.jpeg/.tif/.tiff")

    uploaded = st.file_uploader("Upload images", type=[e[1:] for e in SUPPORTED_EXT], accept_multiple_files=True)
    class_label = st.text_input("Class label to assign (optional)", value="")
    start_btn = st.button("Extract & Append to CSV")

    if start_btn:
        if not uploaded:
            st.warning("Please upload at least one image.")
            return
        rows = []
        with st.spinner("Extracting features..."):
            for file in uploaded:
                dst = TMP_DIR / f"fe_{uuid.uuid4().hex}{Path(file.name).suffix}"
                with open(dst, "wb") as f:
                    f.write(file.getbuffer())
                rows.append(extract_basic_features_from_image(str(dst), class_label.strip() or "unknown"))
        df_new = pd.DataFrame(rows)
        st.subheader("Preview of extracted rows")
        # styled table for nice look
        try:
            st.dataframe(df_new.style.background_gradient(cmap="magma"), height=300)
        except Exception:
            st.dataframe(df_new.head(100))

        # Append or create CSV
        if os.path.exists(CSV_PATH):
            try:
                df_old = pd.read_csv(CSV_PATH)
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df_out = df_new
        else:
            df_out = df_new
        df_out.to_csv(CSV_PATH, index=False)
        st.success(f"Saved/updated CSV at: {CSV_PATH}")
        st.balloons()

def ui_dataset_overview(residuals_obj):
    st.title("Dataset Overview")
    st.write("Explore residuals/fingerprints sets and quick counts.")

    if residuals_obj:
        counts = {}
        for ds in residuals_obj.keys():
            counts[ds] = sum(len(dpi_list) for scanner in residuals_obj[ds].values()
                             for dpi_list in scanner.values())
        st.subheader("Image counts per dataset")
        st.json(counts)

        # Small residuals gallery
        st.subheader("Residual Gallery (first few per dataset)")
        gallery = []
        for ds, sc_map in residuals_obj.items():
            for sc, dpi_map in sc_map.items():
                for dpi, arrs in dpi_map.items():
                    if arrs:
                        gallery.append((ds, residual_to_display(arrs[0])))
                break
        if gallery:
            for i in range(0, len(gallery), 4):
                cols = st.columns(4)
                for j, item in enumerate(gallery[i:i+4]):
                    cols[j].image(item[1], caption=item[0], use_container_width=True)
    else:
        st.info("No residuals pickle loaded. Provide path in the sidebar and reload.")

def ui_feature_visualization(csv_path: str = CSV_PATH):
    st.title("Feature Visualization")
    if not os.path.exists(csv_path):
        st.warning(f"{csv_path} not found. Use Feature Extraction to create it or place it in the project root.")
        return
    try:
        dff = pd.read_csv(csv_path)
        st.subheader("Features table (sample)")
        # show styled dataframe with gradient and zebra
        try:
            styled = dff.head(200).style.background_gradient(cmap="viridis")
            st.dataframe(styled, height=320)
        except Exception:
            st.dataframe(dff.head(200))

        numeric_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Feature distributions")
            feat = st.selectbox("Feature to visualize", numeric_cols, index=0)
            fig, ax = plt.subplots(1, 2, figsize=(14, 4))
            sns.histplot(dff[feat].dropna(), ax=ax[0], kde=True, stat="density", color="#2563eb")
            sns.boxplot(x=dff[feat].dropna(), ax=ax[1], color="#06b6d4")
            ax[0].set_title(f"{feat} distribution")
            ax[1].set_title(f"{feat} boxplot")
            st.pyplot(fig)

            st.subheader("Pairwise scatter (choose two numeric features)")
            pair = st.multiselect("Choose two numeric features", numeric_cols, default=numeric_cols[:2])
            if len(pair) >= 2:
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.scatterplot(x=dff[pair[0]], y=dff[pair[1]], hue=dff.get("class", None), ax=ax2, palette="deep", alpha=0.7)
                ax2.set_xlabel(pair[0]); ax2.set_ylabel(pair[1])
                ax2.set_title(f"{pair[0]} vs {pair[1]}")
                st.pyplot(fig2)

            st.subheader("Feature distributions by class")
            if "class" in dff.columns:
                feat2 = st.selectbox("Feature to compare by class", numeric_cols, index=0, key="feat_by_class")
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                sns.boxplot(x="class", y=feat2, data=dff, ax=ax3, palette="Set2")
                ax3.set_title(f"{feat2} distribution by class")
                st.pyplot(fig3)
            else:
                st.info("No 'class' column found in CSV to group by.")

            st.subheader("Feature correlation matrix")
            corr = dff[numeric_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(12, max(4, 0.3 * len(numeric_cols))))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax4)
            ax4.set_title("Feature correlation matrix")
            st.pyplot(fig4)
        else:
            st.info("No numeric columns detected in CSV.")
    except Exception as e:
        st.error(f"Failed to load/visualize CSV: {e}")

def ui_model_training_eval(residuals_obj, fps, fpks):
    st.title("Model Training & Evaluation")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Baseline models (classical features)")
        if not os.path.exists(CSV_PATH):
            st.warning("official.csv not found. Place your dataset CSV in the app root or specify correct CSV.")
        else:
            if st.button("🚀 Train Baseline RF + SVM"):
                try:
                    with st.spinner("Training baseline models..."):
                        out = train_baseline_models(CSV_PATH, models_dir=str(MODELS_DIR))
                        st.success("Baseline models trained & saved to models/")
                        st.write("RF OOB score:", out.get("oob"))
                        if out.get("feat_imp") is not None:
                            st.subheader("Top feature importances (RF)")
                            fi = out["feat_imp"].reset_index()
                            fi.columns = ["feature", "importance"]
                            st.dataframe(fi.head(30))
                            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(fi.head(30)))))
                            sns.barplot(x="importance", y="feature", data=fi.head(30), ax=ax, palette="rocket")
                            ax.set_title("Top 30 feature importances (RF)")
                            st.pyplot(fig)
                        # saved eval
                        eval_path = MODELS_DIR / "baseline_eval.pkl"
                        if eval_path.exists():
                            try:
                                with open(eval_path, "rb") as f:
                                    ev = pickle.load(f)
                                st.subheader("Baseline evaluation (test split artifact)")
                                st.write("Classes:", ev.get("classes"))
                                cm = ev.get("confusion_matrix")
                                fig2, ax2 = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                                            xticklabels=ev.get("classes"), yticklabels=ev.get("classes"))
                                ax2.set_title("Baseline Confusion Matrix (test split)")
                                st.pyplot(fig2)
                            except Exception:
                                st.info("Saved baseline_eval.pkl exists but couldn't be displayed.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

            st.markdown("**Evaluate saved models**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Evaluate Random Forest"):
                    try:
                        report, cm, classes = evaluate_baseline_model(str(MODELS_DIR / "random_forest.pkl"), CSV_PATH)
                        st.subheader("Classification report (RF)")
                        st.text(report)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=classes, yticklabels=classes)
                        ax.set_title("RF Confusion Matrix (full data)")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"RF evaluation failed: {e}")
            with c2:
                if st.button("Evaluate SVM"):
                    try:
                        report, cm, classes = evaluate_baseline_model(str(MODELS_DIR / "svm.pkl"), CSV_PATH)
                        st.subheader("Classification report (SVM)")
                        st.text(report)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax, xticklabels=classes, yticklabels=classes)
                        ax.set_title("SVM Confusion Matrix (full data)")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"SVM evaluation failed: {e}")

    with col_b:
        st.subheader("CNN / Hybrid quick check")
        model_path_input = st.text_input("Path to trained model (.h5)", value=CNN_MODEL_DEFAULT)
        uploaded_model = st.file_uploader("Or upload a .h5 model (optional)", type=["h5", "keras"], accept_multiple_files=False)
        if uploaded_model is not None:
            tmp_m = Path(f"uploaded_model_{uuid.uuid4().hex}.h5")
            with open(tmp_m, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path_input = str(tmp_m)
            st.success("Model uploaded")

        uploaded_test = st.file_uploader("Upload an image to predict (tif/png/jpg)", type=[e[1:] for e in SUPPORTED_EXT])
        use_example = st.checkbox("Use example residual from loaded pickles", value=(residuals_obj is not None))
        example_choice = None
        if use_example and residuals_obj:
            # keep selectors but make them non-intrusive
            ds_choice = st.selectbox("Dataset", options=list(residuals_obj.keys()))
            sc_choice = st.selectbox("Scanner", options=list(residuals_obj[ds_choice].keys()))
            dpi_choice = st.selectbox("DPI", options=list(residuals_obj[ds_choice][sc_choice].keys()))
            idx_choice = st.number_input("Index", min_value=0,
                                         max_value=max(0, len(residuals_obj[ds_choice][sc_choice][dpi_choice]) - 1),
                                         value=0)
            example_choice = residuals_obj[ds_choice][sc_choice][dpi_choice][idx_choice]

        if st.button("Run Prediction"):
            model = load_tf_model(model_path_input)
            if model is None:
                st.warning("Model not loaded. Provide valid .h5 or upload.")
            else:
                try:
                    res = None
                    if uploaded_test is not None:
                        tmpf = Path("tmp_test_img") / uploaded_test.name
                        tmpf.parent.mkdir(exist_ok=True)
                        with open(tmpf, "wb") as f:
                            f.write(uploaded_test.getbuffer())
                        res = preprocess_residual_pywt_from_image_path(str(tmpf))
                        st.image(residual_to_display(res), caption="Preprocessed residual", use_column_width=True)
                    elif example_choice is not None:
                        res = example_choice
                        st.image(residual_to_display(res), caption="Selected residual example", use_column_width=True)

                    if res is not None:
                        label, conf, preds_raw = infer_from_array(model, res, fps, fpks)
                        st.success(f"✅ Predicted: {label} — Confidence: {conf:.2f}%")
                        # Probability bar chart (with label encoder if present)
                        try:
                            probs = preds_raw
                            le_path = Path("Residuals_Paths") / "hybrid_label_encoder.pkl"
                            if le_path.exists():
                                with open(le_path, "rb") as f:
                                    le = pickle.load(f)
                                labels = list(le.classes_)
                            else:
                                labels = [str(i) for i in range(len(probs))]
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.bar(labels, probs, color=sns.color_palette("magma", len(probs)))
                            ax.set_ylabel("Probability")
                            ax.set_title("Model output probabilities")
                            st.pyplot(fig)
                        except Exception:
                            pass
                    else:
                        st.warning("No image selected for prediction.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

def ui_live_prediction(residuals_obj, fps, fpks):
    st.title("🧪 Live Prediction")
    st.markdown("Predict with **Baseline** (RF/SVM) on raw image or **Hybrid CNN** on residuals+features.")

    mode = st.selectbox("Prediction mode", ["Baseline (RF/SVM)", "Hybrid CNN (image + handcrafted)"])
    uploaded_file = st.file_uploader("Upload an image (tif/png/jpg) to predict", type=[e[1:] for e in SUPPORTED_EXT])
    use_example = st.checkbox("Use example residual from loaded residuals", value=(residuals_obj is not None))
    example_choice = None
    if use_example and residuals_obj:
        ds_choice = st.selectbox("Dataset for example", options=list(residuals_obj.keys()))
        sc_choice = st.selectbox("Scanner", options=list(residuals_obj[ds_choice].keys()))
        dpi_choice = st.selectbox("DPI", options=list(residuals_obj[ds_choice][sc_choice].keys()))
        idx_choice = st.number_input("Index", min_value=0,
                                     max_value=max(0, len(residuals_obj[ds_choice][sc_choice][dpi_choice]) - 1), value=0)
        example_choice = residuals_obj[ds_choice][sc_choice][dpi_choice][idx_choice]

    model = None
    model_path = None
    if mode.startswith("Hybrid"):
        model_path = st.text_input("Path to hybrid .h5 model", value=CNN_MODEL_DEFAULT)
        uploaded_model = st.file_uploader("Upload hybrid model (.h5)", type=["h5", "keras"],
                                          accept_multiple_files=False, key="live_hybrid_model")
        if uploaded_model is not None:
            tmp_m = Path(f"uploaded_live_model_{uuid.uuid4().hex}.h5")
            with open(tmp_m, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = str(tmp_m)
            st.success("Model uploaded")
        if os.path.exists(model_path):
            with st.spinner("Loading hybrid model..."):
                model = load_tf_model(model_path)
            if model:
                st.success("Hybrid model loaded")
            else:
                st.error("Failed to load hybrid model")

    if st.button("Run Prediction"):
        try:
            if uploaded_file is None and example_choice is None:
                st.warning("Upload an image or choose an example residual.")
            else:
                if uploaded_file is not None:
                    tmp_file_path = TMP_DIR / f"tmp_{uuid.uuid4().hex}{Path(uploaded_file.name).suffix}"
                    with open(tmp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if mode.startswith("Baseline"):
                        pred, prob = predict_scanner_baseline(str(tmp_file_path), model_choice="rf")
                        st.image(Image.open(tmp_file_path), caption="Uploaded image", use_column_width=True)
                        st.success(f"Predicted (Baseline): {pred}")
                        if prob is not None:
                            mdl = joblib.load(MODELS_DIR / "random_forest.pkl")
                            classes = mdl.classes_
                            probs = {str(classes[i]): float(round(prob[i], 4)) for i in range(len(classes))}
                            st.subheader("Class probabilities")
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.bar(list(probs.keys()), list(probs.values()), color=sns.color_palette("Set2", len(probs)))
                            ax.set_ylabel("Probability")
                            ax.set_title("Prediction probabilities")
                            st.pyplot(fig)
                            st.json(probs)
                    else:
                        res = preprocess_residual_pywt_from_image_path(str(tmp_file_path))
                        st.image(residual_to_display(res), caption="Residual used for inference")
                        label, conf, _ = infer_from_array(model, res, fps, fpks)
                        st.success(f"Predicted (Hybrid): {label} — Confidence: {conf:.2f}%")
                else:
                    # example residual path
                    res = example_choice
                    if mode.startswith("Baseline"):
                        st.info("Baseline prediction on residuals is not supported directly. Upload raw image for baseline.")
                    else:
                        st.image(residual_to_display(res), caption="Example residual")
                        label, conf, _ = infer_from_array(model, res, fps, fpks)
                        st.success(f"Predicted (Hybrid): {label} — Confidence: {conf:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def ui_about():
    st.title("ℹ️ About")
    st.markdown("""
**Forgery Detection — Unified Streamlit App**

**Features**
- Baseline: Handcrafted feature extraction, RandomForest & SVM training/evaluation.
- CNN / Hybrid: Residual-based CNN inference, fingerprints correlation and hybrid inference.
- Dataset EDA, feature visualization, and live prediction.

**Default file layout**
- `official.csv` (baseline features & labels)
- `Residuals_Paths/official_wiki_residuals.pkl` (residual dataset)
- `Residuals_Paths/scanner_fingerprints.pkl` and `Residuals_Paths/fp_keys.npy`
- `models/` will contain `random_forest.pkl`, `svm.pkl`, `scaler.pkl`

**Tech**: Python, Streamlit, NumPy, pandas, scikit-learn, TensorFlow (optional), OpenCV, PyWavelets, Matplotlib, Seaborn.
""")

# ------------------------ Router ------------------------
if SECTION == "Home":
    ui_home(residuals, fingerprints, fp_keys)
elif SECTION == "Feature Extraction":
    ui_feature_extraction()   # kept (and required)
elif SECTION == "Dataset Overview":
    ui_dataset_overview(residuals)
elif SECTION == "Feature Visualization":
    ui_feature_visualization(CSV_PATH)
elif SECTION == "Model Training & Evaluation":
    ui_model_training_eval(residuals, fingerprints, fp_keys)
elif SECTION == "Live Prediction":
    ui_live_prediction(residuals, fingerprints, fp_keys)
elif SECTION == "About":
    ui_about()

st.markdown("---")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

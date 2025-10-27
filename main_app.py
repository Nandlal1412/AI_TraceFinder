# Imports
import os
import uuid
import pickle
from pathlib import Path
from collections import Counter
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# Baseline ML libs
import cv2
from PIL import Image
import joblib
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# CNN/TF libs
import tensorflow as tf
from tensorflow.keras.models import load_model

# PyWavelets used by CNN preprocess
import pywt

# ---------- CONFIG ----------
st.set_page_config(page_title="Forgery Detection — Combined App", layout="wide")
st.sidebar.title("🔍 App Navigation")
MAIN_PAGE = st.sidebar.radio("Choose app section", [
    "Baseline — Features & Models",
    "CNN — Residuals EDA & Inference",
])

# Shared constants
CSV_PATH = "official.csv"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- UTILITIES ----------
@st.cache_data
def safe_load_pickle(path):
    try:
        if path is None or not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def safe_load_npy(path):
    try:
        if path is None or not os.path.exists(path):
            return None
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

@st.cache_resource
def load_tf_model(path):
    try:
        if path is None or not os.path.exists(path):
            return None
        return load_model(path, compile=False)
    except Exception:
        return None

# ---------- Baseline App Helpers (from baseline_app) ----------

def extract_features(image_path, class_label):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            pil_img = Image.open(image_path).convert("L")
            gray = np.array(pil_img)
        else:
            img = cv2.imread(image_path)
            if img is None:
                return {"file_name": os.path.basename(image_path), "class": class_label, "error": "Unreadable file"}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024
        aspect_ratio = round(width / height, 3) if height != 0 else 0
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3)
        }
    except Exception as e:
        return {"file_name": image_path, "class": class_label, "error": str(e)}


def train_baseline_models(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, MODELS_DIR / "svm.pkl")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")


def evaluate_baseline_model(model_path, name, save_dir="results"):
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader(f"📊 {name} Classification Report")
    st.text(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    st.image(save_path, caption=f"{name} Confusion Matrix", use_column_width=True)


def predict_scanner_baseline(img_path, model_choice="rf"):
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model = joblib.load(MODELS_DIR / ("random_forest.pkl" if model_choice == 'rf' else "svm.pkl"))

    pil_img = Image.open(img_path).convert("L")
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img.shape
    aspect_ratio = w / h if h != 0 else 0
    file_size_kb = os.path.getsize(img_path) / 1024
    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)
    edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    edge_density = np.mean(edges > 0)

    features = pd.DataFrame([{ 
        "width": w, "height": h, "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb, "mean_intensity": mean_intensity,
        "std_intensity": std_intensity, "skewness": skewness,
        "kurtosis": kurt, "entropy": ent, "edge_density": edge_density
    }])

    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob

# ---------- CNN App Helpers (from cnn_app) ----------

IMG_SIZE = (256, 256)

def residual_to_display(res):
    a = res.copy().astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx - mn > 1e-9:
        a = (a - mn) / (mx - mn)
    a = (a * 255).astype(np.uint8)
    return a


def show_image_grid(images, ncols=4, titles=None):
    n = len(images)
    if n == 0:
        st.info("No images to display.")
        return
    nrows = (n + ncols - 1) // ncols
    idx = 0
    for r in range(nrows):
        cols = st.columns(ncols)
        for c in range(ncols):
            if idx >= n:
                break
            with cols[c]:
                img = images[idx]
                if isinstance(img, np.ndarray):
                    arr = img.copy()
                    mn, mx = arr.min(), arr.max()
                    if mx - mn > 1e-9:
                        arr = (arr - mn) / (mx - mn)
                    arr = (arr * 255).astype(np.uint8)
                    st.image(arr, use_container_width=True, caption=(titles[idx] if titles else None))
                else:
                    st.image(img, use_container_width=True, caption=(titles[idx] if titles else None))
            idx += 1


def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0


def preprocess_residual_pywt_from_image_path(path):
    import cv2
    import pywt

    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".tif", ".tiff"]:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError(f"Unable to read {path} as TIFF")
            arr = cv2.resize(arr, IMG_SIZE)
            arr = arr.astype(np.float32) / 255.0
        else:
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=1, dtype=tf.float32)
            img = tf.image.resize(img, IMG_SIZE)
            arr = img.numpy().squeeze()
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")

    cA, (cH, cV, cD) = pywt.dwt2(arr, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    res = (arr - den).astype(np.float32)
    return res

# ---------- App: Baseline Section UI ----------
if MAIN_PAGE == "Baseline — Features & Models":
    st.title("Baseline — Feature Extraction, Training & Prediction")

    page = st.sidebar.radio("Basline Pages", ["Feature Explorer", "Model Evaluation", "Predict Scanner", "Train Models"])

    if page == "Feature Explorer":
        st.header("📁 Feature Extraction Viewer")
        dataset_root = st.text_input("📂 Enter dataset root path:", "")
        if dataset_root and os.path.isdir(dataset_root):
            st.info("🔎 Scanning dataset recursively...")
            records = []
            class_dirs = set()
            for dirpath, _, filenames in os.walk(dataset_root):
                rel_path = os.path.relpath(dirpath, dataset_root)
                if rel_path == ".":
                    continue
                class_name = rel_path.split(os.sep)[0]
                class_dirs.add(class_name)
                img_files = [f for f in filenames if f.lower().endswith(SUPPORTED_EXTENSIONS)]
                for fname in img_files:
                    img_path = os.path.join(dirpath, fname)
                    rec = extract_features(img_path, class_name)
                    records.append(rec)

            if records:
                df = pd.DataFrame(records)
                st.success(f"✅ Detected {len(class_dirs)} classes: {list(class_dirs)}")
                st.dataframe(df.head(20))
                save_path = os.path.join(dataset_root, "metadata_features.csv")
                df.to_csv(save_path, index=False)
                st.success(f"💾 Features saved to {save_path}")
                if "class" in df.columns:
                    st.subheader("📈 Class Distribution")
                    st.bar_chart(df["class"].value_counts())
            else:
                st.warning("⚠️ No supported image files found.")
        elif dataset_root:
            st.error("❌ Invalid dataset path.")

    elif page == "Model Evaluation":
        st.header("📊 Evaluate Trained Models")
        if st.button("Evaluate Random Forest"):
            evaluate_baseline_model(MODELS_DIR / "random_forest.pkl", "Random Forest")
        if st.button("Evaluate SVM"):
            evaluate_baseline_model(MODELS_DIR / "svm.pkl", "SVM")

    elif page == "Predict Scanner":
        st.header("🧪 Predict Document Scanner (Baseline models)")
        uploaded_file = st.file_uploader("Upload a document image", type=list(SUPPORTED_EXTENSIONS))
        model_choice = st.selectbox("Choose model", ["rf", "svm"])
        if uploaded_file:
            temp_path = f"temp_image_{uuid.uuid4().hex}.tif"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            try:
                pred, prob = predict_scanner_baseline(temp_path, model_choice)
                st.success(f"Predicted Scanner: {pred}")
                model = joblib.load(MODELS_DIR / ("random_forest.pkl" if model_choice == 'rf' else "svm.pkl"))
                st.write("Class Probabilities:")
                st.json({cls: round(prob[i], 3) for i, cls in enumerate(model.classes_)})
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif page == "Train Models":
        st.header("🛠️ Train Baseline Models")
        st.markdown("Use this section to retrain your baseline models using `official.csv`.")
        if not os.path.exists(CSV_PATH):
            st.error("❌ Dataset file 'official.csv' not found in project root.")
        else:
            if st.button("🚀 Train Models Now"):
                with st.spinner("Training models..."):
                    try:
                        train_baseline_models()
                        st.success("✅ Models trained and saved successfully!")
                        st.info("Models saved to `models/` folder: - random_forest.pkl - svm.pkl - scaler.pkl")
                    except Exception as e:
                        st.error(f"⚠️ Training failed: {e}")

# ---------- App: CNN Section UI ----------
elif MAIN_PAGE == "CNN — Residuals EDA & Inference":
    st.title("Scanner Residuals — EDA, Fingerprints & Hybrid CNN Inference")
    # Sidebar inputs specific to CNN section
    residuals_pkl = st.sidebar.text_input("Path to official+wiki residuals pickle", value="Residuals_Paths/official_wiki_residuals.pkl")
    flatfield_pkl = st.sidebar.text_input("Path to flatfield residuals pickle (optional)", value="Residuals_Paths/flatfield_residuals.pkl")
    fp_pkl = st.sidebar.text_input("Path to scanner_fingerprints.pkl", value="Residuals_Paths/scanner_fingerprints.pkl")
    fp_keys_npy = st.sidebar.text_input("Path to fp_keys.npy", value="Residuals_Paths/fp_keys.npy")
    model_path_input = st.sidebar.text_input("Path to trained model (.h5)", value="dual_branch_cnn.h5")

    uploaded_model = st.sidebar.file_uploader("Or upload your .h5 model here (optional)", type=["h5", "keras"], accept_multiple_files=False)

    # If model uploaded: save to unique tmp and update model_path_input
    if uploaded_model is not None:
        tmp_m = Path(f"uploaded_model_{uuid.uuid4().hex}.h5")
        with open(tmp_m, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model_path_input = str(tmp_m)
        st.sidebar.success(f"Model uploaded successfully → {uploaded_model.name}")

    # Tabs for CNN content
    dataset_tabs = st.tabs(["Overview", "Official", "Wikipedia", "Flatfield", "Fingerprints & Features", "Inference"])

    # Load pickles
    residuals = safe_load_pickle(residuals_pkl)
    flatfield = safe_load_pickle(flatfield_pkl)
    fingerprints = safe_load_pickle(fp_pkl)
    fp_keys = safe_load_npy(fp_keys_npy)

    # ---------- Overview Tab ----------
    with dataset_tabs[0]:
        st.header("Dataset Overview")
        if residuals is None:
            st.warning(f"Residuals pickle not found at: {residuals_pkl}. Provide a valid path or upload the file.")
        else:
            try:
                total_counts = {}
                for ds in residuals.keys():
                    count = sum(len(dpi_list) for scanner in residuals[ds].values() for dpi_list in scanner.values())
                    total_counts[ds] = count
                df_counts = pd.Series(total_counts).rename_axis('dataset').reset_index(name='count').set_index('dataset')
                st.subheader("Total images by split")
                st.dataframe(df_counts)

                st.subheader("Counts by scanner (combined across dpi)")
                scanner_counter = Counter()
                for ds in residuals.keys():
                    for scanner, dpi_dict in residuals[ds].items():
                        n_imgs = sum(len(lst) for lst in dpi_dict.values())
                        scanner_counter[scanner] += n_imgs
                df_scanner = pd.Series(scanner_counter).rename_axis('scanner').reset_index(name='count').set_index('scanner')
                st.dataframe(df_scanner.sort_values('count', ascending=False))

                st.markdown("**Top 10 scanners (all splits):**")
                st.table(df_scanner.sort_values('count', ascending=False).head(10))

                st.subheader("Scanner distribution (bar plot)")
                # reuse helper
                try:
                    counts = df_scanner.sort_values('count', ascending=False)
                    fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(counts))))
                    sns.barplot(x='count', y='scanner', data=counts.reset_index().head(30), ax=ax)
                    ax.set_title('Images per Scanner (all splits)')
                    st.pyplot(fig)
                except Exception:
                    st.bar_chart(df_scanner.sort_values('count', ascending=False).head(30))

                st.subheader("Show sample residuals")
                select_ds = st.selectbox("Choose dataset", options=list(residuals.keys()), index=0)
                scanners = list(residuals[select_ds].keys())
                sel_scanner = st.selectbox("Choose scanner (show samples)", options=scanners)
                dpis = list(residuals[select_ds][sel_scanner].keys())
                sel_dpi = st.selectbox("DPI folder", options=dpis)
                samples = residuals[select_ds][sel_scanner][sel_dpi]
                n_show = st.slider("Number of samples to preview", min_value=1, max_value=min(32, len(samples)), value=min(8, len(samples)))
                sample_imgs = [residual_to_display(samples[i]) for i in range(n_show)]
                show_image_grid(sample_imgs, ncols=4)
            except Exception as e:
                st.error(f"Failed to render overview: {e}")

    # ---------- Official Tab ----------
    with dataset_tabs[1]:
        st.header("Official — Per-scanner EDA")
        if residuals is None or "Official" not in residuals:
            st.warning("Official dataset residuals not found in provided pickle.")
        else:
            try:
                df = []
                for scanner, dpi_dict in residuals['Official'].items():
                    for dpi, lst in dpi_dict.items():
                        df.append({'scanner': scanner, 'dpi': dpi, 'n_images': len(lst)})
                df_off = pd.DataFrame(df)
                if df_off.empty:
                    st.info("No entries found for Official.")
                else:
                    st.dataframe(df_off)
                    st.subheader("Heatmap: scanner vs dpi (counts)")
                    pivot = df_off.pivot_table(index='scanner', columns='dpi', values='n_images', fill_value=0)
                    fig, ax = plt.subplots(figsize=(10, max(3, 0.25*pivot.shape[0])))
                    sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to render Official tab: {e}")

    # ---------- Wikipedia Tab ----------
    with dataset_tabs[2]:
        st.header("Wikipedia — Per-scanner EDA")
        if residuals is None or "Wikipedia" not in residuals:
            st.warning("Wikipedia dataset residuals not found in provided pickle.")
        else:
            try:
                df = []
                for scanner, dpi_dict in residuals['Wikipedia'].items():
                    for dpi, lst in dpi_dict.items():
                        df.append({'scanner': scanner, 'dpi': dpi, 'n_images': len(lst)})
                df_wiki = pd.DataFrame(df)
                if df_wiki.empty:
                    st.info("No entries found for Wikipedia.")
                else:
                    st.dataframe(df_wiki)
                    st.subheader("Distribution: images per scanner")
                    counts = df_wiki.groupby('scanner')['n_images'].sum().sort_values(ascending=False)
                    # reuse plotting helper
                    fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(counts))))
                    sns.barplot(x=counts.values, y=counts.index, ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to render Wikipedia tab: {e}")

    # ---------- Flatfield Tab ----------
    with dataset_tabs[3]:
        st.header("Flatfield Residuals & Fingerprints")
        if flatfield is None:
            st.info("Flatfield residuals pickle not provided or not found. If you have one, set path in sidebar or upload file.")
        else:
            try:
                scanners = list(flatfield.keys())
                st.markdown(f"Found flatfield scanners: {scanners[:20]}")
                sel_sc = st.selectbox("Select flatfield scanner", options=scanners)
                flat_samples = flatfield[sel_sc][:min(12, len(flatfield[sel_sc]))]
                show_image_grid([residual_to_display(r) for r in flat_samples], ncols=4)
            except Exception as e:
                st.error(f"Failed to render Flatfield tab: {e}")

    # ---------- Fingerprints & Features Tab ----------
    with dataset_tabs[4]:
        st.header("Fingerprints & Feature Previews")
        if fingerprints is None or fp_keys is None:
            st.warning("Fingerprints or fp_keys not found. Provide paths in the sidebar or upload files.")
        else:
            try:
                st.subheader("Available fingerprints")
                fp_list = list(fingerprints.keys())
                st.write(fp_list[:50])
                sel_fp = st.selectbox("Choose fingerprint to visualize", options=fp_list)
                fp_img = residual_to_display(fingerprints[sel_fp])
                st.image(fp_img, caption=f"Fingerprint: {sel_fp}")

                st.subheader("Compare residual -> fingerprint correlation (example)")
                if residuals and 'Official' in residuals:
                    _any = None
                    for sc, dpi_dict in residuals['Official'].items():
                        for dpi, lst in dpi_dict.items():
                            if len(lst) > 0:
                                _any = lst[0]; break
                        if _any is not None: break
                    if _any is not None:
                        try:
                            key_list = list(fp_keys) if fp_keys is not None else list(fingerprints.keys())
                        except Exception:
                            key_list = list(fingerprints.keys())
                        corr_vals = {k: corr2d(_any, fingerprints[k]) for k in key_list}
                        corr_ser = pd.Series(corr_vals).sort_values(ascending=False)
                        st.write(corr_ser.head(10))
                        fig, ax = plt.subplots(figsize=(8,4))
                        sns.barplot(x=corr_ser.values[:20], y=corr_ser.index[:20], ax=ax)
                        ax.set_title('ZNCC with top 20 fingerprints')
                        st.pyplot(fig)
                    else:
                        st.info("No sample residual available to correlate with fingerprints.")
                else:
                    st.info("Official residuals not available for correlation example.")
            except Exception as e:
                st.error(f"Failed to render Fingerprints tab: {e}")

    # ---------- Inference Tab ----------
    with dataset_tabs[5]:
        st.header("Inference — Use your trained hybrid model")
        st.markdown("Load model and run prediction on uploaded or example images. (scaler.pkl has been removed — inference uses image branch and dummy handcrafted features if needed)")

        model = load_tf_model(model_path_input)

        if model is None:
            st.warning("Model not loaded. Upload or provide a valid .h5 model path in the sidebar.")
        else:
            st.success("Model loaded successfully")

        col1, col2 = st.columns([2,1])
        example_choice = None
        with col1:
            st.subheader("Select test image")
            uploaded_test = st.file_uploader("Upload an image to predict (tif/png/jpg)", type=["tif","tiff","png","jpg","jpeg"])
            use_example = st.checkbox("Or use example image from dataset", value=(residuals is not None))
            if use_example and residuals is not None:
                ds_choice = st.selectbox("Dataset for example image", options=list(residuals.keys()))
                sc_choice = st.selectbox("Scanner for example image", options=list(residuals[ds_choice].keys()))
                dpi_choice = st.selectbox("DPI for example image", options=list(residuals[ds_choice][sc_choice].keys()))
                idx_choice = st.number_input("Index within that folder", min_value=0, max_value=max(0, len(residuals[ds_choice][sc_choice][dpi_choice]) - 1), value=0)
                example_choice = residuals[ds_choice][sc_choice][dpi_choice][idx_choice]

        with col2:
            st.subheader("Model info")
            if model is not None:
                try:
                    layer_info = [f"{i}: {layer.__class__.__name__} — output shape {getattr(layer, 'output_shape', 'unknown')}" for i, layer in enumerate(model.layers)]
                    st.text('\n'.join(layer_info))
                except Exception:
                    st.text("Layer information unavailable for this model.")

        def make_feats_from_res_for_infer(res, fingerprints, fp_keys_local):
            try:
                keys = list(fp_keys_local)
            except Exception:
                keys = list(fingerprints.keys())
            v_corr = [corr2d(res, fingerprints[k]) for k in keys]
            def fft_radial_energy(img, K=6):
                f = np.fft.fftshift(np.fft.fft2(img))
                mag = np.abs(f)
                h, w = mag.shape; cy, cx = h//2, w//2
                yy, xx = np.ogrid[:h, :w]
                r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
                rmax = r.max() + 1e-6
                bins = np.linspace(0, rmax, K+1)
                feats = []
                for i in range(K):
                    m = (r >= bins[i]) & (r < bins[i+1])
                    feats.append(float(mag[m].mean() if m.any() else 0.0))
                return feats
            v_fft = fft_radial_energy(res, K=6)
            v_lbp = [0.0]*10
            return np.array(v_corr + v_fft + v_lbp, dtype=np.float32)

        def infer_from_array(res_arr):
            if model is None:
                raise RuntimeError("Model not loaded")

            x_img = np.expand_dims(res_arr, axis=(0, -1)).astype(np.float32)

            handcrafted_ready = (fingerprints is not None) and (fp_keys is not None)
            if not handcrafted_ready:
                st.info("Fingerprints or fp_keys missing — running image-branch only with dummy handcrafted features.")
                feats = np.zeros((1, 32), dtype=np.float32)
            else:
                feats_raw = make_feats_from_res_for_infer(res_arr, fingerprints, fp_keys.tolist())
                feats = feats_raw.reshape(1, -1).astype(np.float32)

            try:
                n_inputs = len(model.inputs)

                if n_inputs == 1:
                    preds = model.predict(x_img, verbose=0)

                elif n_inputs == 2:
                    input_shapes = [tuple(inp.shape) for inp in model.inputs]
                    if any(s == (None, 256, 256, 1) for s in input_shapes):
                        if input_shapes[0] == (None, 256, 256, 1):
                            preds = model.predict([x_img, feats], verbose=0)
                        else:
                            preds = model.predict([feats, x_img], verbose=0)
                    else:
                        preds = model.predict([x_img, feats], verbose=0)
                else:
                    raise ValueError(f"Unexpected number of model inputs: {n_inputs}")

                idx = int(np.argmax(preds[0]))
                conf = float(np.max(preds[0]) * 100.0)

                label = str(idx)
                le_path = Path("Residuals_Paths") / "hybrid_label_encoder.pkl"
                if le_path.exists():
                    with open(le_path, 'rb') as f:
                        le = pickle.load(f)
                    if hasattr(le, "classes_") and idx < len(le.classes_):
                        label = le.classes_[idx]

                return label, conf

            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                return None, None

        if st.button("Run Prediction"):
            try:
                res = None
                if uploaded_test is not None:
                    tmp = Path("tmp_test_img")
                    tmp.mkdir(exist_ok=True)
                    tmpf = tmp / uploaded_test.name
                    with open(tmpf, 'wb') as f:
                        f.write(uploaded_test.getbuffer())
                    res = preprocess_residual_pywt_from_image_path(str(tmpf))
                    disp = residual_to_display(res)
                    st.image(disp, caption="Preprocessed residual")

                elif example_choice is not None:
                    res = example_choice
                    disp = residual_to_display(res)
                    st.image(disp, caption="Selected example residual")

                if res is not None:
                    label, conf = infer_from_array(res)
                    if label is not None:
                        st.success(f"✅ **Predicted:** {label} — **Confidence:** {conf:.2f}%")
                    else:
                        st.error("Prediction failed — label could not be determined.")
                else:
                    st.warning("No image selected. Upload an image or choose an example from dataset.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption("Combined Streamlit app: Baseline features + CNN residuals. Ensure required files exist in the project folder before using respective sections.")

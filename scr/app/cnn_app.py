# streamlit_scanner_app.py
# Streamlit app to explore scanner residual dataset, flatfield fingerprints,
# visualize EDA, preview sample images, and run inference with your trained hybrid CNN.
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run streamlit_scanner_app.py
#
# Required files:
#  - Residuals_Paths/official_wiki_residuals.pkl
#  - Residuals_Paths/flatfield_residuals.pkl  (optional)
#  - Residuals_Paths/scanner_fingerprints.pkl
#  - Residuals_Paths/fp_keys.npy
#  - dual_branch_cnn.h5  (your trained model)

import os
import pickle
from pathlib import Path
from collections import Counter
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ML libs
import tensorflow as tf
from tensorflow.keras.models import load_model

# Helpers
st.set_page_config(layout="wide", page_title="Scanner Residuals EDA & Inference")

# ---------- Utility functions ----------
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

# Visualization helpers
def plot_counts_table(data, title="Counts"):
    if isinstance(data, pd.Series):
        df = data.rename_axis('category').reset_index(name='count')
    elif isinstance(data, pd.DataFrame):
        if 'count' in data.columns and 'category' in data.columns:
            df = data[['category', 'count']]
        elif data.shape[1] == 1:
            df = data.reset_index()
            df.columns = ['category', 'count']
        else:
            st.error("DataFrame missing 'count' column — cannot plot.")
            return
    else:
        st.error("Unsupported data type for plot_counts_table.")
        return

    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(df))))
    sns.barplot(x='count', y='category', data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

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

def residual_to_display(res):
    a = res.copy().astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx - mn > 1e-9:
        a = (a - mn) / (mx - mn)
    a = (a * 255).astype(np.uint8)
    return a

# Prediction helpers
import pywt
IMG_SIZE = (256, 256)

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

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

# ---------- App layout ----------
st.title("Scanner Residuals — EDA, Visualization & Hybrid CNN Inference")
st.markdown("Upload or point to your dataset & trained models. The app will visualize dataset composition, sample residuals, fingerprints, and let you run inference using your hybrid CNN.")

# Sidebar: file inputs
st.sidebar.header("Data & Models")
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

# Tabs
dataset_tabs = st.tabs(["Overview", "Official", "Wikipedia", "Flatfield", "Fingerprints & Features", "Inference"])

# Load pickles (after upload handling)
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
        # Build counts across splits
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
            plot_counts_table(df_scanner.sort_values('count', ascending=False), title="Images per Scanner (all splits)")

            st.subheader("Show sample residuals")
            # choose dataset
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
                plot_counts_table(counts, title="Wikipedia: Images per Scanner")
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
            # sample one residual from Official if available
            if residuals and 'Official' in residuals:
                _any = None
                for sc, dpi_dict in residuals['Official'].items():
                    for dpi, lst in dpi_dict.items():
                        if len(lst) > 0:
                            _any = lst[0]; break
                    if _any is not None: break
                if _any is not None:
                    # ensure fp_keys is an iterable of fingerprint keys
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

    # Layout for selecting test image
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
            # show a compact layer summary
            try:
                layer_info = [f"{i}: {layer.__class__.__name__} — output shape {getattr(layer, 'output_shape', 'unknown')}" for i, layer in enumerate(model.layers)]
                st.text('\n'.join(layer_info))
            except Exception:
                st.text("Layer information unavailable for this model.")

    # Feature creation for inference (match training pipeline)
    def make_feats_from_res_for_infer(res, fingerprints, fp_keys_local):
        # safe fp_keys_local -> list of keys
        try:
            keys = list(fp_keys_local)
        except Exception:
            keys = list(fingerprints.keys())
        v_corr = [corr2d(res, fingerprints[k]) for k in keys]
        # FFT radial energy
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

    # Prepare CNN input
    x_img = np.expand_dims(res_arr, axis=(0, -1)).astype(np.float32)

    # Prepare handcrafted features (no scaler)
    handcrafted_ready = (fingerprints is not None) and (fp_keys is not None)
    if not handcrafted_ready:
        st.info("Fingerprints or fp_keys missing — running image-branch only with dummy handcrafted features.")
        feats = np.zeros((1, 32), dtype=np.float32)  # default to 32 dummy features
    else:
        feats_raw = make_feats_from_res_for_infer(res_arr, fingerprints, fp_keys.tolist())
        feats = feats_raw.reshape(1, -1).astype(np.float32)

    try:
        # Determine model input structure safely
        n_inputs = len(model.inputs)

        if n_inputs == 1:
            preds = model.predict(x_img, verbose=0)

        elif n_inputs == 2:
            # Each inp.shape may already be tuple-like
            input_shapes = [tuple(inp.shape) for inp in model.inputs]

            # Detect which input expects the image
            if any(s == (None, 256, 256, 1) for s in input_shapes):
                if input_shapes[0] == (None, 256, 256, 1):
                    preds = model.predict([x_img, feats], verbose=0)
                else:
                    preds = model.predict([feats, x_img], verbose=0)
            else:
                # Fallback: assume image-first
                preds = model.predict([x_img, feats], verbose=0)

        else:
            raise ValueError(f"Unexpected number of model inputs: {n_inputs}")

        # Get top prediction
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100.0)

        # Try decoding label safely
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


# --- Prediction button section ---
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

# Footer
st.markdown("---")
st.caption("Streamlit app created to visualize residual datasets, fingerprints and run inference with a hybrid CNN. Scaler logic removed as requested.")

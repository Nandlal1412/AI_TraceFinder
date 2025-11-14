# cnn app

import streamlit as st
st.set_page_config(page_title="AI TraceFinder — Forensic Scanner Identification", layout="wide")

import os, io, sys, pickle, subprocess, tempfile, glob
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import base64
import time

# --- Helper: safe imports for ML/TensorFlow ---
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern
except Exception:
    local_binary_pattern = None

# -------------------------
# Utilities (re-implemented safely)
# -------------------------
IMG_SIZE = (256,256)

def safe_read_image(path, as_gray=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if as_gray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize_img_for_residual(img):
    # handle 8-bit and 16-bit gracefully
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        # fallback
        m = float(img.max() or 255.0)
        return img.astype(np.float32) / m

def wavelet_denoise_residual(img):
    # use pywt if available, else simple blur
    try:
        import pywt
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH[:] = 0; cV[:] = 0; cD[:] = 0
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)
    except Exception:
        # fallback: gaussian blur
        den = cv2.GaussianBlur(img, (5,5), 0)
        return (img - den).astype(np.float32)

def preprocess_to_residual(path):
    img = safe_read_image(path, as_gray=True)
    if img is None:
        raise ValueError("Unable to read image.")
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    imgf = normalize_img_for_residual(img)
    res = wavelet_denoise_residual(imgf)
    return res

def corr2d(a,b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h,w = mag.shape; cy,cx = h//2, w//2
    yy,xx = np.ogrid[:h,:w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.any():
            feats.append(float(mag[mask].mean()))
        else:
            feats.append(0.0)
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    if local_binary_pattern is None:
        # fallback basic histogram of intensities
        hist,_ = np.histogram(img, bins=8, range=(0,1), density=True)
        return hist.tolist()
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

# -------------
# File helpers
# -------------
ROOT = Path(".")
DATA_DIR = ROOT / "Datasets"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_pickle_safe(p):
    try:
        with open(p,"rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

def list_dataset_structure(base=DATA_DIR):
    rows = []
    if not base.exists():
        return pd.DataFrame(rows, columns=["scanner","dpi_or_file","count"])
    for scanner in sorted([d for d in base.iterdir() if d.is_dir()]):
        # if scanner contains dpi folders:
        subdirs = [d for d in scanner.iterdir() if d.is_dir()]
        if subdirs:
            for dpi in subdirs:
                files = list(dpi.glob("*.*"))
                rows.append([scanner.name, dpi.name, sum(1 for f in files if f.suffix.lower() in [".tif",".tiff",".png",".jpg",".jpeg"])])
        else:
            files = list(scanner.glob("*.*"))
            rows.append([scanner.name, ".", sum(1 for f in files if f.suffix.lower() in [".tif",".tiff",".png",".jpg",".jpeg"])])
    return pd.DataFrame(rows, columns=["scanner","dpi_or_file","count"])

# ---------------------------
# Streamlit UI per page
# ---------------------------
st.sidebar.title("AI TraceFinder")
page = st.sidebar.radio("Navigate", ["Home","Dataset Overview","Feature Visualization","EDA","Feature Extraction","Model Training & Evaluation","Live Prediction","About"])

# ---------- HOME ----------
if page == "Home":
    st.markdown("# 🧠 AI TraceFinder — Forensic Scanner Identification")
    st.write("Detecting document forgery by analyzing unique scanner fingerprints using hybrid CNN and handcrafted features.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ⚙️ How It Works")
        st.write("""
        **Pipeline:** Input → Preprocessing → Residual Extraction → Feature Fusion (PRNU, FFT, LBP) → Hybrid CNN Model → Prediction  
        """)
        st.markdown("### 🧩 Tech Stack")
        st.write("- Python • Streamlit • TensorFlow/Keras • scikit-learn • OpenCV • PyWavelets")
    
    # --- Dataset Summary Section ---
    st.markdown("### 📂 Dataset Information")
    total_residuals = 0
    dataset_summary = []

    for ds_name in ["Official", "Wikipedia", "Flatfield"]:
        pkl_path = DATA_DIR / f"{ds_name.lower()}_residuals.pkl" if ds_name == "Flatfield" else DATA_DIR / "official_wiki_residuals.pkl"
        if ds_name in ["Official", "Wikipedia"]:
            # Load only once for both
            if ds_name == "Official" and pkl_path.exists():
                rd = load_pickle_safe(pkl_path)
                for sub_ds in ["Official", "Wikipedia"]:
                    if sub_ds in rd:
                        scanners = list(rd[sub_ds].keys())
                        res_count = sum(len(r) for dpi_dict in rd[sub_ds].values() for r in dpi_dict.values())
                        total_residuals += res_count
                        dataset_summary.append((sub_ds, len(scanners), res_count))
        elif ds_name == "Flatfield" and pkl_path.exists():
            rd_flat = load_pickle_safe(pkl_path)
            scanners = list(rd_flat.keys())
            res_count = sum(len(v) for v in rd_flat.values())
            total_residuals += res_count
            dataset_summary.append((ds_name, len(scanners), res_count))

    if dataset_summary:
        df_info = pd.DataFrame(dataset_summary, columns=["Dataset", "Scanners", "Residual Images"])
        st.dataframe(df_info, use_container_width=True)
        st.metric("Total Residuals Across All Datasets", f"{total_residuals:,}")
    else:
        st.warning("No residual data found. Please run `processing.py` to generate dataset residuals.")

    with col2:
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=120)

# ---------- DATASET OVERVIEW ----------
elif page == "Dataset Overview":
    st.header("📂 Dataset Overview")
    st.write("This section summarizes the datasets used to train and evaluate the Hybrid CNN model.")

    # Load dataset structure summary
    df = list_dataset_structure(DATA_DIR)
    if df.empty:
        st.warning("No dataset found in the Datasets/ folder.")
    else:
        st.dataframe(df, use_container_width=True)

        # --- Compute summary metrics ---
        total_scanners = df['scanner'].nunique()
        total_images = int(df['count'].sum())
        avg_per_scanner = total_images / total_scanners if total_scanners else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scanners", total_scanners)
        col2.metric("Total Images", f"{total_images:,}")
        col3.metric("Avg. Images per Scanner", f"{avg_per_scanner:.1f}")

        # --- Dataset composition from processing output ---
        st.markdown("### 📊 Dataset Composition (Official, Wikipedia, Flatfield)")
        summary_rows = []
        total_res = 0
        for ds_name, file_name in {
            "Official/Wikipedia": "official_wiki_residuals.pkl",
            "Flatfield": "flatfield_residuals.pkl"
        }.items():
            pkl_path = DATA_DIR / file_name
            if pkl_path.exists():
                rd = load_pickle_safe(pkl_path)
                if rd:
                    if ds_name == "Official/Wikipedia":
                        for sub in rd.keys():
                            scanners = list(rd[sub].keys())
                            res_count = sum(len(r) for dpi_dict in rd[sub].values() for r in dpi_dict.values())
                            total_res += res_count
                            summary_rows.append((sub, len(scanners), res_count))
                    else:
                        scanners = list(rd.keys())
                        res_count = sum(len(r) for r in rd.values())
                        total_res += res_count
                        summary_rows.append((ds_name, len(scanners), res_count))
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows, columns=["Dataset", "Scanners", "Residuals"])
            st.dataframe(df_sum, use_container_width=True)

            # --- Pie chart for dataset contribution ---
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(df_sum["Residuals"], labels=df_sum["Dataset"], autopct="%1.1f%%", startangle=90)
            ax.set_title("Residual Distribution Across Datasets", fontsize=13)
            st.pyplot(fig)

        # --- Bar chart: per-scanner image distribution ---
        st.markdown("### 🔍 Scanner-wise Image Distribution")
        scanner_counts = df.groupby("scanner")["count"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(scanner_counts.index, scanner_counts.values)
        ax.set_ylabel("Image Count")
        ax.set_xlabel("Scanner Model")
        ax.set_title("Images per Scanner")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # --- DPI Distribution ---
        st.markdown("### 📈 DPI Distribution")
        dpi_counts = df.groupby("dpi_or_file")["count"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(dpi_counts.index, dpi_counts.values, color="skyblue")
        ax.set_xlabel("Image Count")
        ax.set_ylabel("DPI / Category")
        ax.set_title("Distribution of Images by DPI")
        st.pyplot(fig)

        st.info("Use this overview to verify dataset balance and distribution before feature extraction or training.")

# ---------- FEATURE VISUALIZATION ----------
elif page == "Feature Visualization":
    st.header("🧩 Feature Visualization")
    st.write("Visual exploration of handcrafted features (PRNU, FFT, LBP) extracted for scanner classification.")

    feat_files = {
        "PRNU features (features.pkl)": MODELS_DIR / "features.pkl",
        "Enhanced features (enhanced_features.pkl)": MODELS_DIR / "enhanced_features.pkl"
    }
    choice = st.selectbox("Select available feature file", list(feat_files.keys()))
    fpath = feat_files[choice]
    data = load_pickle_safe(fpath) if fpath.exists() else None

    if data is None:
        st.warning(f"Feature file not found: {fpath}")
    else:
        feats = np.array(data.get("features"))
        labels = np.array(data.get("labels"))
        st.write(f"Loaded features: **{feats.shape[0]} samples × {feats.shape[1]} features**")

        # ---- Histogram for a selected feature ----
        st.subheader("📈 Feature Distribution (Histogram)")
        num_show = min(10, feats.shape[1])
        sel_feat = st.slider("Select feature index", 0, feats.shape[1]-1, 0)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(feats[:, sel_feat], bins=40, color="skyblue", edgecolor="black")
        ax.set_title(f"Feature #{sel_feat} Distribution")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # ---- Correlation Heatmap ----
        with st.expander("🔍 Feature Correlation Heatmap"):
            import seaborn as sns
            corr_subset = pd.DataFrame(feats[:, :min(20, feats.shape[1])])
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr_subset.corr(), cmap="coolwarm", ax=ax)
            ax.set_title("Correlation (first 20 features)")
            st.pyplot(fig)

        # ---- Feature variance ranking ----
        with st.expander("📊 Feature Variance Ranking"):
            var_vals = np.var(feats, axis=0)
            idxs = np.argsort(var_vals)[::-1][:10]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(range(10), var_vals[idxs], color="teal")
            ax.set_xticks(range(10))
            ax.set_xticklabels([f"f{i}" for i in idxs], rotation=45)
            ax.set_ylabel("Variance")
            ax.set_title("Top 10 Most Variable Features")
            st.pyplot(fig)

        # ---- PCA Scatter (2D) ----
        with st.expander("🧭 PCA Projection (2D)"):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            proj = pca.fit_transform(feats)
            df_proj = pd.DataFrame({"x": proj[:,0], "y": proj[:,1], "label": labels})
            fig, ax = plt.subplots(figsize=(7,5))
            for lab in np.unique(labels):
                sub = df_proj[df_proj.label == lab]
                ax.scatter(sub.x, sub.y, label=str(lab), s=12, alpha=0.7)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.legend(fontsize="small", ncol=2)
            ax.set_title("PCA — Feature Clusters by Scanner")
            st.pyplot(fig)

        # ---- Per-class radar plot ----
        with st.expander("🕸️ Class-wise Feature Centroids (Radar Chart)"):
            try:
                import plotly.graph_objects as go
                df_feat = pd.DataFrame(feats)
                df_feat["label"] = labels
                mean_df = df_feat.groupby("label").mean()
                feat_labels = [f"f{i}" for i in range(min(10, feats.shape[1]))]
                fig = go.Figure()
                for lab in mean_df.index:
                    vals = mean_df.loc[lab, :len(feat_labels)].values.tolist()
                    vals.append(vals[0])  # close the loop
                    fig.add_trace(go.Scatterpolar(r=vals, theta=feat_labels,
                                                  fill='toself', name=str(lab)))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                  showlegend=True, title="Feature Centroid Profiles")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Radar plot could not be generated: {e}")

# ---------- EDA ----------
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("Dataset & features statistical summaries.")
    # Show precomputed residuals stats if present
    res_path = DATA_DIR / "official_wiki_residuals.pkl"
    if res_path.exists():
        st.success("Found residuals pickle: official_wiki_residuals.pkl")
        rd = load_pickle_safe(res_path)
        # quick stats: number of scanners and total residual images
        total = 0
        scanners = set()
        for dataset_name in rd.keys():
            for scanner, dpi_dict in rd[dataset_name].items():
                scanners.add(scanner)
                for dpi, res_list in dpi_dict.items():
                    total += len(res_list)
        st.write(f"Datasets present: {list(rd.keys())}")
        st.write(f"Unique scanners (across datasets): {len(scanners)}")
        st.write(f"Total residual images found: {total}")

        # ---- Visualize pre-saved EDA charts (if available) ----
        eda_dir = ROOT / "results" / "eda_charts"
        if eda_dir.exists():
            st.markdown("### 🖼️ Visual EDA Charts & Insights")
            img_files = sorted([p for p in eda_dir.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            if img_files:
                n_cols = 3  # number of images per row
                for i in range(0, len(img_files), n_cols):
                    cols = st.columns(n_cols)
                    for j, img_path in enumerate(img_files[i:i+n_cols]):
                        with cols[j]:
                            st.image(
                                str(img_path),
                                caption=img_path.stem.replace("_", " ").title(),
                                use_container_width=True
                            )
            else:
                st.info("No EDA chart images found in `results/eda_charts/`. Place your .png/.jpg files there to view them.")
        else:
            st.info("EDA charts folder not found. Create a `results/eda_charts/` directory in your project and add your analysis images.")

    else:
        st.warning("Residuals pickle not found. Run processing.py to generate residuals first.")
    # Allow quick EDA on feature files
    fpath = MODELS_DIR / "enhanced_features.pkl"
    if fpath.exists():
        ef = load_pickle_safe(fpath)
        feats = np.array(ef["features"])
        labels = np.array(ef["labels"])
        st.write("Enhanced features shape:", feats.shape)
        stats = pd.DataFrame(feats).describe().T
        st.dataframe(stats.head(20))
        # correlation heatmap (first 30 features)
        import seaborn as sns
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        subset = pd.DataFrame(feats[:,:min(30,feats.shape[1])])
        sns.heatmap(subset.corr(), ax=ax)
        ax.set_title("Feature correlation (first 30 features)")
        st.pyplot(fig)
    else:
        st.info("enhanced_features.pkl not available under models/")

# ---------- FEATURE EXTRACTION ----------
elif page == "Feature Extraction":
    st.header("⚙️ Feature Extraction")
    st.write("Extract PRNU, FFT, and LBP-based features either from uploaded images or from your dataset folders.")

    option = st.radio("Select extraction mode:", ["🔹 Upload Images", "📁 Extract from Dataset Folder"])

    # -------------------------
    # MODE 1 — MANUAL IMAGE UPLOAD
    # -------------------------
    if option == "🔹 Upload Images":
        uploaded = st.file_uploader("Upload one or multiple images", accept_multiple_files=True, type=["png","jpg","jpeg","tif","tiff"])
        if uploaded:
            rows = []
            for u in uploaded:
                bytes_data = u.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(u.name).suffix) as tmp:
                    tmp.write(bytes_data)
                    tmp_path = tmp.name
                try:
                    res = preprocess_to_residual(tmp_path)

                    # --- Enhanced Residual Visualization (brighter + contrast stretch)
                    res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)
                    res_enhanced = np.clip(res_norm * 2.2, 0, 1)
                    res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[..., None], 3, axis=2)
                    st.image(res_rgb, caption=f"Enhanced Residual — {u.name}", use_container_width=False)

                    # --- Compute Features
                    fp = load_pickle_safe(MODELS_DIR / "scanner_fingerprints.pkl")
                    fp_keys = None
                    if fp:
                        fp_keys = np.load(MODELS_DIR / "fp_keys.npy", allow_pickle=True).tolist()
                        v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                    else:
                        v_corr = []
                    v_fft = fft_radial_energy(res, K=6)
                    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
                    feat_vec = v_corr + v_fft + v_lbp

                    rows.append({
                        "filename": u.name,
                        "corr_len": len(v_corr),
                        "fft_len": len(v_fft),
                        "lbp_len": len(v_lbp),
                        "feature_vector": feat_vec
                    })
                except Exception as e:
                    st.error(f"❌ Failed for {u.name}: {e}")

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df[["filename","corr_len","fft_len","lbp_len"]])
                if st.button("Download features as CSV"):
                    df2 = df.copy()
                    df2["feature_vector"] = df2["feature_vector"].apply(lambda x: ",".join(map(str, x)))
                    csv = df2.to_csv(index=False)
                    st.download_button("Download CSV", data=csv.encode(), file_name="uploaded_features.csv", mime="text/csv")

    # -------------------------
    # MODE 2 — AUTOMATIC DATASET EXTRACTION
    # -------------------------
    elif option == "📁 Extract from Dataset Folder":
        st.info("This will extract features for **all images** found in your dataset structure under `Datasets/`. Large datasets may take time.")
        dataset_base = st.text_input("Enter dataset path (default: ./Datasets)", value=str(DATA_DIR))

        if st.button("Start Dataset Feature Extraction"):
            dataset_path = Path(dataset_base)
            if not dataset_path.exists():
                st.error(f"❌ Dataset path not found: {dataset_path}")
            else:
                all_images = [p for p in dataset_path.rglob("*") if p.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff"]]
                total_imgs = len(all_images)
                if total_imgs == 0:
                    st.warning("No images found in the dataset folder.")
                else:
                    rows = []
                    progress = st.progress(0)
                    sample_images = np.random.choice(all_images, size=min(5, total_imgs), replace=False)
                    st.write(f"Found **{total_imgs}** images. Extracting features...")

                    for i, img_path in enumerate(all_images, 1):
                        try:
                            res = preprocess_to_residual(str(img_path))

                            # --- Only preview a few random residuals ---
                            if img_path in sample_images:
                                res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)
                                res_enhanced = np.clip(res_norm * 2.2, 0, 1)
                                res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[..., None], 3, axis=2)
                                st.image(res_rgb, caption=f"Residual Sample — {img_path.name}", use_container_width=True)

                            # --- Compute Features ---
                            fp = load_pickle_safe(MODELS_DIR / "scanner_fingerprints.pkl")
                            fp_keys = None
                            if fp:
                                fp_keys = np.load(MODELS_DIR / "fp_keys.npy", allow_pickle=True).tolist()
                                v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                            else:
                                v_corr = []
                            v_fft = fft_radial_energy(res, K=6)
                            v_lbp = lbp_hist_safe(res, P=8, R=1.0)
                            feat_vec = v_corr + v_fft + v_lbp

                            scanner_name = img_path.parent.name
                            rows.append({
                                "filepath": str(img_path),
                                "scanner": scanner_name,
                                "feature_vector": feat_vec
                            })
                        except Exception as e:
                            st.warning(f"⚠️ Skipped {img_path.name}: {e}")
                        progress.progress(i / total_imgs)

                    # --- Save and Preview Results ---
                    if rows:
                        df = pd.DataFrame(rows)
                        st.success(f"✅ Feature extraction complete — processed {len(df)} images.")
                        st.dataframe(df.head(10))

                        save_path = MODELS_DIR / "dataset_extracted_features.csv"
                        df2 = df.copy()
                        df2["feature_vector"] = df2["feature_vector"].apply(lambda x: ",".join(map(str, x)))
                        df2.to_csv(save_path, index=False)
                        st.download_button(
                            "📥 Download Dataset Features (CSV)",
                            data=df2.to_csv(index=False).encode(),
                            file_name="dataset_extracted_features.csv",
                            mime="text/csv"
                        )

# ---------- MODEL TRAINING & EVALUATION ----------
elif page == "Model Training & Evaluation":
    st.header("🧠 Model Training & Evaluation")
    st.write("Evaluate pre-trained Hybrid CNN models using stored features or training history files.")

    # --- Cached Model Loader ---
    @st.cache_resource(show_spinner=False)
    def load_tf_model_cached(path):
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)

    # --- Model selection dropdown ---
    model_files = {
        "Hybrid CNN (Final)": MODELS_DIR / "scanner_hybrid_final.keras",
        "Hybrid CNN (Checkpoint)": MODELS_DIR / "scanner_hybrid.keras"
    }
    selected_model_label = st.selectbox("Select a model to evaluate", list(model_files.keys()))
    model_path = model_files[selected_model_label]

    encoder_path = MODELS_DIR / "hybrid_label_encoder.pkl"
    scaler_path = MODELS_DIR / "hybrid_feat_scaler.pkl"
    feature_files = {
        "Enhanced Features": MODELS_DIR / "enhanced_features.pkl",
        "Baseline Features": MODELS_DIR / "features.pkl"
    }
    selected_feat_label = st.selectbox("Select feature data", list(feature_files.keys()))
    feature_path = feature_files[selected_feat_label]

    hist_path = MODELS_DIR / "hybrid_training_history.pkl"

    # --- Model existence ---
    if not model_path.exists():
        st.error(f"❌ Model not found: {model_path.name}")
        st.stop()
    st.success(f"✅ Model ready: {model_path.name}")

    if TF_AVAILABLE:
        model = load_tf_model_cached(str(model_path))
        st.caption(f"Model has **{len(model.layers)} layers**, inputs: {[tuple(x.shape) for x in model.inputs]}")
    else:
        st.warning("TensorFlow not available — cannot evaluate.")
        st.stop()

    # --- Optional training curves ---
    if hist_path.exists():
        with st.expander("📉 Training History Overview"):
            hist = load_pickle_safe(hist_path)
            if hist:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(hist.get("accuracy", []), label="Train Acc")
                ax[0].plot(hist.get("val_accuracy", []), label="Val Acc")
                ax[0].legend(); ax[0].set_title("Accuracy")
                ax[1].plot(hist.get("loss", []), label="Train Loss")
                ax[1].plot(hist.get("val_loss", []), label="Val Loss")
                ax[1].legend(); ax[1].set_title("Loss")
                st.pyplot(fig)

    # --- Preprocessors ---
    encoder = load_pickle_safe(encoder_path)
    scaler = load_pickle_safe(scaler_path)
    if encoder is None:
        st.warning("⚠️ Label encoder missing.")
    if scaler is None:
        st.warning("⚠️ Feature scaler missing.")

    # --- Load features efficiently ---
    if not feature_path.exists():
        st.warning("Feature file missing — cannot evaluate.")
        st.stop()

    ef = load_pickle_safe(feature_path)
    feats = np.array(ef["features"], dtype=np.float32)
    labels = np.array(ef["labels"])

    st.write(f"Loaded features: {feats.shape[0]} samples × {feats.shape[1]} features")

    # --- Option: limit evaluation sample size ---
    max_samples = st.slider("Limit evaluation to N samples", 100, feats.shape[0], min(1000, feats.shape[0]), 100)
    idx = np.random.choice(feats.shape[0], max_samples, replace=False)
    feats, labels = feats[idx], labels[idx]

    # --- Scale if applicable ---
    if scaler is not None and getattr(scaler, "n_features_in_", feats.shape[1]) == feats.shape[1]:
        feats = scaler.transform(feats)
    else:
        st.caption("Skipping scaling due to mismatch.")

    # --- Align dimensions with model ---
    expected_dim = int(model.inputs[1].shape[1])
    if feats.shape[1] != expected_dim:
        feats = feats[:, :expected_dim] if feats.shape[1] > expected_dim else np.pad(
            feats, ((0, 0), (0, expected_dim - feats.shape[1])), mode='constant'
        )

    dummy_img = np.zeros((feats.shape[0], 256, 256, 1), dtype=np.float32)

    # --- Predict in batches for efficiency ---
    st.info("Running fast batched inference...")
    batch_size = 256
    preds_list = []
    for i in range(0, len(feats), batch_size):
        batch_feats = feats[i:i+batch_size]
        batch_imgs = dummy_img[i:i+batch_size]
        preds_list.append(model.predict([batch_imgs, batch_feats], verbose=0))
    preds = np.vstack(preds_list)

    y_pred = np.argmax(preds, axis=1)
    y_true = encoder.transform(labels)

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    report = classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(encoder.classes_)*0.5), 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Predicted vs True Scanners", fontsize=14)
    st.pyplot(fig)

    # --- Per-class accuracy chart ---
    with st.expander("📈 Class-wise Accuracy Overview"):
        accs = [report[c]["recall"] for c in encoder.classes_ if c in report]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(encoder.classes_, accs, color="teal")
        ax.set_ylabel("Recall / Accuracy")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ---------- LIVE PREDICTION ----------
elif page == "Live Prediction":
    st.header("Live Prediction")
    st.write("Upload an image and pick a model to predict the scanner.")
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","tif","tiff"])
    model_files = {
        "Hybrid CNN (scanner_hybrid_final.keras)": MODELS_DIR / "scanner_hybrid_final.keras",
        "Hybrid CNN (scanner_hybrid.keras)": MODELS_DIR / "scanner_hybrid.keras"
    }
    chosen_model = st.selectbox("Model file", list(model_files.keys()))
    if uploaded and st.button("Predict"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.getvalue()); tmp.flush()
        tmp.close()
        model_path = model_files[chosen_model]
        if not model_path.exists():
            st.error(f"Model not found: {model_path}")
        else:
            try:
                # preprocess
                res = preprocess_to_residual(tmp.name)
                x_img = np.expand_dims(res, axis=(0,-1)).astype(np.float32)
                # features
                fp = load_pickle_safe(MODELS_DIR / "scanner_fingerprints.pkl")
                fp_keys = None
                if fp:
                    fp_keys = np.load(MODELS_DIR / "fp_keys.npy", allow_pickle=True).tolist()
                    v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                else:
                    v_corr = []
                v_fft = fft_radial_energy(res, K=6)
                v_lbp = lbp_hist_safe(res, P=8, R=1.0)
                feat_vec = np.array([v_corr + v_fft + v_lbp], dtype=np.float32)
                # scale features
                scaler = load_pickle_safe(MODELS_DIR / "hybrid_feat_scaler.pkl")
                if scaler is not None:
                    feat_vec = scaler.transform(feat_vec)
                # load model
                if TF_AVAILABLE:
                    model = tf.keras.models.load_model(str(model_path))
                    prob = model.predict([x_img, feat_vec], verbose=0)
                    idx = int(np.argmax(prob, axis=1)[0])
                    encoder = load_pickle_safe(MODELS_DIR / "hybrid_label_encoder.pkl")
                    label = encoder.classes_[idx] if encoder is not None else str(idx)
                    conf = float(prob[0, idx]*100)
                    st.write("**Prediction:**", label)
                    st.write("**Confidence:** %.2f%%" % conf)
                    # show residual image
                    # --- Enhanced residual visualization (bright, clear, and Streamlit-safe) ---
                    # Normalize to 0–1 for consistent contrast
                    res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)

                    # Apply slight contrast stretching to make fingerprint patterns visible
                    res_enhanced = np.clip(res_norm * 1.5, 0, 1)

                    # Convert to 3-channel (for consistent display)
                    res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[..., None], 3, axis=2)

                    # Display using updated Streamlit API
                    st.image(res_rgb, caption="Enhanced Residual Preview", use_container_width=False)

                else:
                    st.error("TensorFlow not available in environment. Install TF to run hybrid model predictions.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- ABOUT ----------
elif page == "About":
    st.header("ℹ️ About — AI TraceFinder")
    st.markdown("""
    **AI TraceFinder** is a forensic machine learning platform for **scanner source identification**.  
    It detects document forgeries by analyzing unique **scanner noise fingerprints** (PRNU patterns) and texture statistics.

    ---
    ### 🧠 Core Idea
    Every scanner leaves subtle sensor and mechanical patterns during image acquisition.  
    TraceFinder captures these patterns using:
    - **PRNU Correlation (Fingerprint Similarity)**
    - **FFT-based Frequency Energy Distribution**
    - **LBP-based Microtexture Analysis**

    These handcrafted features are fused with a **Hybrid CNN** that learns scanner-specific patterns from denoised residual images.

    ---
    ### ⚙️ Current Setup
    | Model | Description |
    |:--|:--|
    | 🧩 **CNN_Model** | Deep-learning branch for residual-based classification |
    | 🌲 **Random Forest_Model** | Feature-based baseline for handcrafted features |
    | 🔹 **SVM_Model** | Traditional baseline classifier for quick evaluation |

    ---
    ### 🧩 Technologies Used
    - **Frontend/UI:** Streamlit  
    - **Image Processing:** OpenCV, PyWavelets  
    - **Data Handling:** NumPy, Pandas  
    - **Machine Learning:** TensorFlow / scikit-learn  
    - **Visualization:** Matplotlib, Seaborn, Plotly

    ---
    ### 👥 Team
    **Lead Developer:** Your Name  
    **Institution:** XYZ  
    **Contact:** your.email@example.com  

    ---
    ### 🔗 Resources
    - 🧾 [GitHub Repository](https://github.com/your-repo-link)
    - 📘 Project Report: `AI_TraceFinder.pdf`

    ---
    ### 🚀 Deployment Notes
    - Replace dummy models with your **trained versions** in the `/models/` folder.
    - Ensure your **datasets** (`Official`, `Wikipedia`, `Flatfield`) are under `/Datasets/`.
    - The app auto-detects and integrates your `.pkl`, `.keras`, and `.csv` artifacts.

    ---
    **AI TraceFinder** — Empowering Digital Forensics with Machine Learning.
    """)

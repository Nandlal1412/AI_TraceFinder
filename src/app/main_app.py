import streamlit as st
st.set_page_config(page_title="AI TraceFinder — Forensic Scanner Identification", layout="wide")

# Standard imports used across both apps
import os
import io
import sys
import tempfile
import glob
import base64
from pathlib import Path
import pickle
import joblib
import time

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# ML / utilities
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# optional imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern
except Exception:
    local_binary_pattern = None

# Optional plotting libs
try:
    import seaborn as sns
except Exception:
    sns = None

# ------------------
# Configuration & Paths
# ------------------
ROOT = Path('.')
DATA_DIR = ROOT / 'Datasets'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = 'official.csv'
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
IMG_SIZE = (256, 256)

# ------------------
# Helper functions 
# ------------------

def load_pickle_safe(p):
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def safe_read_image(path, as_gray=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if as_gray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize_img_for_residual(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        m = float(img.max() or 255.0)
        return img.astype(np.float32) / m

# wavelet denoise residual (pywt if available, else gaussian)
def wavelet_denoise_residual(img):
    try:
        import pywt
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH[:] = 0; cV[:] = 0; cD[:] = 0
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)
    except Exception:
        den = cv2.GaussianBlur(img, (5,5), 0)
        return (img - den).astype(np.float32)

def preprocess_to_residual(path):
    img = safe_read_image(path, as_gray=True)
    if img is None:
        raise ValueError('Unable to read image.')
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    imgf = normalize_img_for_residual(img)
    res = wavelet_denoise_residual(imgf)
    return res

# correlation and spectral features
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
        hist,_ = np.histogram(img, bins=8, range=(0,1), density=True)
        return hist.tolist()
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method='uniform')
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

# Baseline feature extraction (from baseline_app)
def extract_features(image_path, class_label):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.tif', '.tiff']:
            pil_img = Image.open(image_path).convert('L')
            gray = np.array(pil_img)
        else:
            img = cv2.imread(image_path)
            if img is None:
                return {'file_name': os.path.basename(image_path), 'class': class_label, 'error': 'Unreadable file'}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024
        aspect_ratio = round(width / height, 3)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            'file_name': os.path.basename(image_path),
            'class': class_label,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'file_size_kb': round(file_size, 2),
            'mean_intensity': round(mean_intensity, 3),
            'std_intensity': round(std_intensity, 3),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurt, 3),
            'entropy': round(shannon_entropy, 3),
            'edge_density': round(edge_density, 3)
        }
    except Exception as e:
        return {'file_name': image_path, 'class': class_label, 'error': str(e)}

# Baseline model training (from baseline_app)
def train_baseline_models():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'CSV dataset not found: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['file_name', 'class'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, MODELS_DIR / 'svm.pkl')

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

# Baseline evaluation (from baseline_app)
def evaluate_baseline_model(model_path, name, save_dir='results'):
    if not os.path.exists(CSV_PATH):
        st.error('Dataset CSV not found for evaluation.')
        return
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['file_name', 'class'])
    y = df['class']
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if not scaler_path.exists():
        st.error('Baseline scaler missing (models/scaler.pkl). Train baseline models first.')
        return
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader(f'📊 {name} Classification Report')
    st.text(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8,6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', ax=ax)
    else:
        ax.imshow(cm, interpolation='nearest')
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    st.image(save_path, caption=f'{name} Confusion Matrix', use_column_width=True)

# Baseline predict (from baseline_app)
def predict_baseline_scanner(img_path, model_choice='rf'):
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError('Baseline scaler missing (models/scaler.pkl)')
    scaler = joblib.load(scaler_path)
    model_file = 'random_forest.pkl' if model_choice == 'rf' else 'svm.pkl'
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f'Baseline model missing: {model_path}')
    model = joblib.load(model_path)

    pil_img = Image.open(img_path).convert('L')
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w = img.shape
    aspect_ratio = w / h
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
        'width': w, 'height': h, 'aspect_ratio': aspect_ratio,
        'file_size_kb': file_size_kb, 'mean_intensity': mean_intensity,
        'std_intensity': std_intensity, 'skewness': skewness,
        'kurtosis': kurt, 'entropy': ent, 'edge_density': edge_density
    }])

    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob, model.classes_

# Dataset structure listing (from cnn_app)
def list_dataset_structure(base=DATA_DIR):
    rows = []
    if not base.exists():
        return pd.DataFrame(rows, columns=['scanner','dpi_or_file','count'])
    for scanner in sorted([d for d in base.iterdir() if d.is_dir()]):
        subdirs = [d for d in scanner.iterdir() if d.is_dir()]
        if subdirs:
            for dpi in subdirs:
                files = list(dpi.glob('*.*'))
                rows.append([scanner.name, dpi.name, sum(1 for f in files if f.suffix.lower() in SUPPORTED_EXT)])
        else:
            files = list(scanner.glob('*.*'))
            rows.append([scanner.name, '.', sum(1 for f in files if f.suffix.lower() in SUPPORTED_EXT)])
    return pd.DataFrame(rows, columns=['scanner','dpi_or_file','count'])

# ------------------
# UI: exact sidebar with 8 pages 
# ------------------
st.sidebar.title('AI TraceFinder')
page = st.sidebar.radio('Navigate', [
    'Home',
    'Dataset Overview',
    'Feature Visualization',
    'EDA',
    'Feature Extraction',
    'Modal Training & Evaluation',
    'Live Prediction',
    'Forgery / Tamper Detection',
    'About'
])

# ---------- HOME ----------
if page == 'Home':

    # ----------- HEADER CARD -----------
    st.markdown("""
        <div style="padding: 25px; border-radius: 15px; 
                    background-color: #f7f9fc; border: 1px solid #e6e9ef;">
            <h1 style="color:#2c3e50; margin-bottom: 5px;">
                🧠 AI TraceFinder — Forensic Scanner Identification
            </h1>
            <p style="font-size:18px; color:#34495e;">
                Detecting document forgery by analyzing unique scanner fingerprints 
                using Hybrid CNN and handcrafted feature fusion.
            </p>
        </div>
        <br>
    """, unsafe_allow_html=True)

    # Layout columns
    col1, col2 = st.columns([2, 1], gap="large")

    # ----------- HOW IT WORKS CARD -----------
    with col1:
        st.markdown("""
            <div style="padding: 20px; border-radius: 12px; 
                        background-color: #ffffff; border: 1px solid #eaeaea;">
                <h3 style="color:#2c3e50;">⚙️ How It Works</h3>
                <p style="font-size:16px; color:#4e5d6c;">
                    <b>Pipeline:</b><br>
                    Input → Preprocessing → Residual Extraction → Feature Fusion 
                    (PRNU, FFT, LBP) → Hybrid CNN Model → Prediction
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------- TECH STACK CARD -----------
        st.markdown("""
            <div style="padding: 20px; border-radius: 12px; 
                        background-color: #ffffff; border: 1px solid #eaeaea;">
                <h3 style="color:#2c3e50;">🧩 Tech Stack</h3>
                <ul style="color:#4e5d6c; font-size:15px; line-height:1.6;">
                    <li>Python</li>
                    <li>Streamlit</li>
                    <li>TensorFlow / Keras</li>
                    <li>scikit-learn</li>
                    <li>OpenCV</li>
                    <li>PyWavelets</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # ----------- RIGHT COLUMN (LOGO CARD) -----------
    with col2:
        st.markdown("""
            <div style="text-align:center; padding: 20px; 
                        background-color: #ffffff; border-radius: 12px; 
                        border:1px solid #eaeaea;">
                <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" 
                    width="140">
                <p style="color:#4e5d6c; font-size:14px; margin-top:10px;">
                    Powered by Streamlit
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ----------- DATASET INFORMATION HEADING -----------
    st.markdown("<br><h3 style='color:#2c3e50;'>📂 Dataset Information</h3>", unsafe_allow_html=True)

    # ----------- DATASET SUMMARY LOGIC -----------
    total_residuals = 0
    dataset_summary = []

    for ds_name in ['Official', 'Wikipedia', 'Flatfield']:
        pkl_path = DATA_DIR / f"{ds_name.lower()}_residuals.pkl" if ds_name == 'Flatfield' else DATA_DIR / 'official_wiki_residuals.pkl'

        if ds_name in ['Official', 'Wikipedia']:
            if ds_name == 'Official' and pkl_path.exists():
                rd = load_pickle_safe(pkl_path)
                for sub_ds in ['Official', 'Wikipedia']:
                    if rd and sub_ds in rd:
                        scanners = list(rd[sub_ds].keys())
                        res_count = sum(len(r) for dpi_dict in rd[sub_ds].values() for r in dpi_dict.values())
                        total_residuals += res_count
                        dataset_summary.append((sub_ds, len(scanners), res_count))

        elif ds_name == 'Flatfield' and pkl_path.exists():
            rd_flat = load_pickle_safe(pkl_path)
            if rd_flat:
                scanners = list(rd_flat.keys())
                res_count = sum(len(v) for v in rd_flat.values())
                total_residuals += res_count
                dataset_summary.append((ds_name, len(scanners), res_count))

    # ----------- COLORFUL TABLE + METRIC CARD -----------
    if dataset_summary:

        df_info = pd.DataFrame(dataset_summary, 
                               columns=['Dataset','Scanners','Residual Images'])

        st.markdown("""
            <p style="color:#4e5d6c; font-size:15px; margin-bottom:10px;">
                Summary of available datasets and extracted residual fingerprints.
            </p>
        """, unsafe_allow_html=True)

        # ---- Color Styling for DataFrame ----
        def style_table(df):
            return (
                df.style
                .set_properties(**{
                    "background-color": "#ffffff",
                    "border": "1px solid #eaeaea",
                    "color": "#2c3e50",
                    "padding": "8px"
                })
                .set_table_styles([
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#e8f1ff"),
                            ("color", "#2c3e50"),
                            ("font-weight", "bold"),
                            ("border", "1px solid #d3def2"),
                        ]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #f0f0f0"),
                        ]
                    }
                ])
                .bar(subset=["Residual Images"], 
                     color="#b3d1ff",
                     vmax=df["Residual Images"].max())
            )

        st.dataframe(
            style_table(df_info),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------- COLORFUL METRIC CARD -----------
        st.markdown(f"""
            <div style="
                padding: 16px;
                border-radius: 12px;
                background: linear-gradient(135deg, #e8f1ff 0%, #bcd9ff 100%);
                border: 1px solid #d2e4ff;
                width: 260px;
            ">
                <h4 style="color:#2c3e50; margin: 0;">Total Residuals</h4>
                <p style="font-size:24px; color:#1b3c70; margin: 5px 0 0;">
                    {total_residuals:,}
                </p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.warning('No residual data found. Please run processing.py to generate dataset residuals.')

# ---------- DATASET OVERVIEW ----------
elif page == 'Dataset Overview':

    # ---- Header Card ----
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px;
                    background-color: #f7f9fc; border: 1px solid #e6e9ef; margin-bottom: 20px;">
            <h2 style="color:#2c3e50; margin: 0;">📂 Dataset Overview</h2>
            <p style="font-size:16px; color:#4e5d6c; margin-top:8px;">
                Explore the datasets used by AI TraceFinder. Counts below are computed from the residual
                pickle artifacts (more reliable than a single folder walk).
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ---- Compute accurate per-dataset counts from residual pickles ----
    official_pickle = DATA_DIR / "official_wiki_residuals.pkl"
    flatfield_pickle = DATA_DIR / "flatfield_residuals.pkl"

    ms = {
        "Official": {"scanners": 0, "images": 0},
        "Wikipedia": {"scanners": 0, "images": 0},
        "Flatfield": {"scanners": 0, "images": 0}
    }

    # Official / Wikipedia
    if official_pickle.exists():
        rd = load_pickle_safe(official_pickle) or {}
        for subset in ["Official", "Wikipedia"]:
            if subset in rd and isinstance(rd[subset], dict):
                subset_dict = rd[subset]
                ms[subset]["scanners"] = len(subset_dict)
                img_count = 0
                for scanner_name, dpi_dict in subset_dict.items():
                    if isinstance(dpi_dict, dict):
                        for dpi, res_list in dpi_dict.items():
                            img_count += len(res_list or [])
                    else:
                        img_count += len(dpi_dict or [])
                ms[subset]["images"] = img_count

    # Flatfield
    if flatfield_pickle.exists():
        rd_flat = load_pickle_safe(flatfield_pickle) or {}
        ms["Flatfield"]["scanners"] = len(rd_flat)
        ms["Flatfield"]["images"] = sum(len(v or []) for v in rd_flat.values())

    # --------------------------------------------------------------------
    # 📌 MOVE CARDS TO THE TOP 
    # --------------------------------------------------------------------
    col_off, col_wiki, col_flat = st.columns(3, gap="large")

    col_off.markdown(f"""
        <div style="padding:14px;border-radius:12px;
                    background:linear-gradient(135deg,#e8f1ff,#bcd9ff);border:1px solid #d2e4ff;">
            <h4 style="color:#2c3e50;margin:0;">📘 Official Scanners</h4>
            <p style="font-size:22px;color:#1b3c70;margin:6px 0 0;">{ms['Official']['scanners']}</p>
            <small style="color:#4e5d6c;">Images: {ms['Official']['images']:,}</small>
        </div>
    """, unsafe_allow_html=True)

    col_wiki.markdown(f"""
        <div style="padding:14px;border-radius:12px;
                    background:linear-gradient(135deg,#dff0ff,#c6e9ff);border:1px solid #cfe8ff;">
            <h4 style="color:#2c3e50;margin:0;">📗 Wikipedia Scanners</h4>
            <p style="font-size:22px;color:#1b3c70;margin:6px 0 0;">{ms['Wikipedia']['scanners']}</p>
            <small style="color:#4e5d6c;">Images: {ms['Wikipedia']['images']:,}</small>
        </div>
    """, unsafe_allow_html=True)

    col_flat.markdown(f"""
        <div style="padding:14px;border-radius:12px;
                    background:linear-gradient(135deg,#f5fbff,#e2f1ff);border:1px solid #d9eaff;">
            <h4 style="color:#2c3e50;margin:0;">📙 Flatfield Scanners</h4>
            <p style="font-size:22px;color:#1b3c70;margin:6px 0 0;">{ms['Flatfield']['scanners']}</p>
            <small style="color:#4e5d6c;">Images: {ms['Flatfield']['images']:,}</small>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # 📌 NOW Residual Dataset Summary table
    # --------------------------------------------------------------------
    res_rows = [(k, ms[k]["scanners"], ms[k]["images"]) for k in ["Official", "Wikipedia", "Flatfield"]]
    df_res = pd.DataFrame(res_rows, columns=["Dataset", "Scanners", "Residual Images"])

    st.markdown("<h3 style='color:#2c3e50;'>📊 Residual Dataset Summary</h3>", unsafe_allow_html=True)
    st.dataframe(
        df_res.style.set_properties(**{"color": "#2c3e50"}).bar(subset=["Residual Images"], color="#b3d1ff"),
        use_container_width=True,
        hide_index=True
    )

    # ---- Pie chart
    if df_res["Residual Images"].sum() > 0:
        fig, ax = plt.subplots(figsize=(4,4))
        colors = ["#a8c6ff", "#cfe0ff", "#8fb4ff"]

        ax.pie(
            df_res["Residual Images"],
            labels=df_res["Dataset"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"linewidth":1,"edgecolor":"white"},
            textprops={"color":"#2c3e50","fontsize":11}
        )
        ax.set_title("Residual Distribution Across Datasets", color="#2c3e50", fontsize=13)
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # 📌 Folder-scan table 
    # --------------------------------------------------------------------
    df = list_dataset_structure(DATA_DIR)
    if df.empty:
        st.warning("No dataset found in the Datasets/ folder.")
        st.stop()
    else:
        def style_dataset_table(df):
            return (df.style
                    .set_properties(**{"background-color":"#ffffff","border":"1px solid #eaeaea",
                                       "color":"#2c3e50","padding":"8px"})
                    .set_table_styles([{"selector":"th",
                                        "props":[("background-color","#e8f1ff"),
                                                 ("color","#2c3e50"),
                                                 ("font-weight","bold"),
                                                 ("border","1px solid #d3def2")]}]))

        st.markdown("<h4 style='color:#2c3e50;'>📁 Folder Scan (quick view)</h4>", unsafe_allow_html=True)
        st.dataframe(style_dataset_table(df), use_container_width=True, hide_index=True)

    # ---- Scanner-wise distribution
    st.markdown("<h3 style='color:#2c3e50;'>🔍 Scanner-wise Image Distribution (folder scan)</h3>", unsafe_allow_html=True)
    try:
        scanner_counts = df.groupby("scanner", as_index=False)["count"].sum().sort_values("count", ascending=False)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(scanner_counts["scanner"], scanner_counts["count"], color="#a7c7ff")
        ax.set_ylabel("Image Count", color="#2c3e50")
        ax.set_xlabel("Scanner (folder-level)", color="#2c3e50")
        ax.set_title("Images per Scanner (folder scan)", color="#2c3e50")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    except:
        st.warning("Could not draw scanner-wise folder distribution.")

    # ---- DPI distribution
    st.markdown("<h3 style='color:#2c3e50;'>📈 DPI / Subfolder Distribution</h3>", unsafe_allow_html=True)
    try:
        dpi_group = df.groupby("dpi_or_file", as_index=False)["count"].sum().sort_values("count")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(dpi_group["dpi_or_file"], dpi_group["count"], color="#bcd9ff")
        ax.set_xlabel("Image Count", color="#2c3e50")
        ax.set_ylabel("DPI / Subfolder", color="#2c3e50")
        ax.set_title("Distribution by DPI / Subfolder", color="#2c3e50")
        st.pyplot(fig)
    except:
        st.info("No DPI/subfolder breakdown available.")

    st.info("Displayed counts come from residual pickles (preferred). Folder-scan table is provided for debugging and quick inspection.")

# ---------- FEATURE VISUALIZATION ----------
elif page == 'Feature Visualization':

    # Header Card
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px;
                    background-color: #f7f9fc; border: 1px solid #e6e9ef; margin-bottom: 20px;">
            <h2 style="color:#2c3e50; margin: 0;">🧩 Feature Visualization</h2>
            <p style="font-size:16px; color:#4e5d6c; margin-top:8px;">
                Visual exploration of handcrafted feature vectors (PRNU, FFT, LBP) extracted from scanner images.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature file selection
    st.markdown("<h4 style='color:#2c3e50;'>📁 Select Feature File</h4>", unsafe_allow_html=True)

    feat_files = {
        'PRNU features (features.pkl)': MODELS_DIR / 'features.pkl',
        'Enhanced features (enhanced_features.pkl)': MODELS_DIR / 'enhanced_features.pkl'
    }
    choice = st.selectbox('', list(feat_files.keys()))
    fpath = feat_files[choice]
    data = load_pickle_safe(fpath) if fpath.exists() else None

    if data is None:
        st.warning(f'Feature file not found: {fpath}')
    else:
        feats = np.array(data.get('features'))
        labels = np.array(data.get('labels'))

        # Info Card
        st.markdown(f"""
            <div style="padding:16px;border-radius:12px;margin-top:10px;
                        background:linear-gradient(135deg,#e8f1ff,#bcd9ff);
                        border:1px solid #d2e4ff;">
                <h4 style="color:#2c3e50;margin:0;">📦 Loaded Feature Matrix</h4>
                <p style="color:#1b3c70;font-size:18px;margin:4px 0 0;">
                    {feats.shape[0]} samples × {feats.shape[1]} features
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Histogram Section
        st.markdown("<br><h4 style='color:#2c3e50;'>📈 Feature Value Distribution</h4>", unsafe_allow_html=True)
        sel_feat = st.slider('Select feature index', 0, feats.shape[1]-1, 0)

        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(feats[:, sel_feat], bins=40, edgecolor='white', color="#a8c6ff")
        ax.set_title(f'Feature #{sel_feat} Distribution', color="#2c3e50")
        ax.set_xlabel('Feature Value', color="#2c3e50")
        ax.set_ylabel('Frequency', color="#2c3e50")
        fig.patch.set_facecolor('#ffffff')
        st.pyplot(fig)

        # Heatmap
        with st.expander('📊 Feature Correlation Heatmap (Top 20 Features)'):
            corr_subset = pd.DataFrame(feats[:, :min(20, feats.shape[1])])
            fig, ax = plt.subplots(figsize=(8,6))

            if sns is not None:
                sns.heatmap(
                    corr_subset.corr(),
                    cmap=sns.light_palette("#7ea0ff", as_cmap=True),
                    ax=ax,
                    annot=False,
                    cbar=True
                )
            else:
                ax.imshow(corr_subset.corr(), cmap="Blues")

            ax.set_title("Correlation Between Top Features", color="#2c3e50")
            st.pyplot(fig)

        # Variance Ranking
        with st.expander('📈 Top 10 Most Variable Features'):
            var_vals = np.var(feats, axis=0)
            idxs = np.argsort(var_vals)[::-1][:10]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(range(10), var_vals[idxs], color="#8fb4ff")
            ax.set_xticks(range(10))
            ax.set_xticklabels([f'f{i}' for i in idxs], rotation=45)
            ax.set_ylabel('Variance', color="#2c3e50")
            ax.set_title('Top 10 Most Variable Features', color="#2c3e50")
            st.pyplot(fig)

        # PCA Projection
        with st.expander('🧭 PCA 2D Projection'):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            proj = pca.fit_transform(feats)
            df_proj = pd.DataFrame({'x': proj[:,0], 'y': proj[:,1], 'label': labels})

            fig, ax = plt.subplots(figsize=(7,5))
            colors = ["#a8c6ff", "#8fb4ff", "#c9d8ff", "#dfeaff", "#bcd9ff"]

            for i, lab in enumerate(np.unique(labels)):
                sub = df_proj[df_proj.label == lab]
                ax.scatter(sub.x, sub.y, label=str(lab), s=18, alpha=0.7, color=colors[i % len(colors)])

            ax.set_xlabel('PC1', color="#2c3e50")
            ax.set_ylabel('PC2', color="#2c3e50")
            ax.set_title('PCA: Feature Clustering by Scanner', color="#2c3e50")
            ax.legend(fontsize='small', ncol=2)
            st.pyplot(fig)

        # Radar Chart
        with st.expander('🕸️ Scanner-wise Feature Centroids (Radar Chart)'):
            try:
                import plotly.graph_objects as go
                df_feat = pd.DataFrame(feats)
                df_feat['label'] = labels
                mean_df = df_feat.groupby('label').mean()

                feat_labels = [f'f{i}' for i in range(min(10, feats.shape[1]))]

                fig = go.Figure()

                pastel = ["#a8c6ff", "#8fb4ff", "#c9d8ff", "#dfeaff", "#bcd9ff"]

                for i, lab in enumerate(mean_df.index):
                    vals = mean_df.loc[lab, :len(feat_labels)].values.tolist()
                    vals.append(vals[0])
                    fig.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=feat_labels,
                        fill='toself',
                        name=str(lab),
                        line=dict(color=pastel[i % len(pastel)])
                    ))

                fig.update_layout(
                    title="Feature Centroid Profiles",
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Radar plot could not be generated: {e}")

# ---------- EDA ----------
elif page == 'EDA':

    # Header card
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px;
                    background-color: #f7f9fc; border: 1px solid #e6e9ef; margin-bottom: 20px;">
            <h2 style="color:#2c3e50; margin: 0;">📊 Exploratory Data Analysis</h2>
            <p style="font-size:16px; color:#4e5d6c; margin-top:8px;">
                Statistical summaries, correlations, and visual insights extracted from scanner residual datasets.
            </p>
        </div>
    """, unsafe_allow_html=True)

    res_path = DATA_DIR / 'official_wiki_residuals.pkl'

    # ----------------------------------------------------------------
    # Residual dataset stats (Official + Wikipedia)
    # ----------------------------------------------------------------
    if res_path.exists():
        st.success('✔ Found residuals pickle: official_wiki_residuals.pkl')

        rd = load_pickle_safe(res_path)
        total = 0
        scanners = set()

        for dataset_name in (rd.keys() if rd else []):
            for scanner, dpi_dict in rd[dataset_name].items():
                scanners.add(scanner)
                for dpi, res_list in dpi_dict.items():
                    total += len(res_list)

        # Summary Cards
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#e8f1ff,#bcd9ff);
                        border:1px solid #d2e4ff;">
                <h4 style="color:#2c3e50;margin:0;">📁 Datasets Loaded</h4>
                <p style="font-size:20px;color:#1b3c70;margin:4px 0;">
                    {', '.join(rd.keys())}
                </p>
            </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#dff0ff,#c6e9ff);
                        border:1px solid #cfe8ff;">
                <h4 style="color:#2c3e50;margin:0;">🖨️ Unique Scanners</h4>
                <p style="font-size:22px;color:#1b3c70;margin:4px 0;">
                    {len(scanners)}
                </p>
            </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#f5fbff,#e2f1ff);
                        border:1px solid #d9eaff;">
                <h4 style="color:#2c3e50;margin:0;">🖼️ Total Residual Images</h4>
                <p style="font-size:22px;color:#1b3c70;margin:4px 0;">
                    {total:,}
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # Display EDA image files (heatmaps, charts, etc.)
        # ----------------------------------------------------------------
        eda_dir = ROOT / 'results' / 'eda_charts'
        st.markdown("<h3 style='color:#2c3e50;'>🖼️ Visual EDA Charts</h3>", unsafe_allow_html=True)

        if eda_dir.exists():
            img_files = sorted([p for p in eda_dir.glob('*.*') 
                                if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

            if img_files:
                n_cols = 3
                for i in range(0, len(img_files), n_cols):
                    cols = st.columns(n_cols)
                    for j, img_path in enumerate(img_files[i:i+n_cols]):
                        with cols[j]:
                            st.image(
                                str(img_path),
                                caption=img_path.stem.replace('_', ' ').title(),
                                use_container_width=True
                            )
            else:
                st.info('No EDA chart images found in `results/eda_charts/`.')
        else:
            st.info('⚠️ No EDA charts folder found. Create `results/eda_charts/` and add your EDA images.')

    else:
        st.warning('⚠️ Residuals pickle not found. Run processing.py to generate residuals first.')

    # ----------------------------------------------------------------
    # Enhanced Features Summary + Correlation Heatmap
    # ----------------------------------------------------------------
    st.markdown("<br><h3 style='color:#2c3e50;'>📈 Enhanced Feature (LBP + FFT + PRNU) Statistics</h3>", unsafe_allow_html=True)

    fpath = MODELS_DIR / 'enhanced_features.pkl'

    if fpath.exists():
        ef = load_pickle_safe(fpath)
        feats = np.array(ef['features'])
        labels = np.array(ef['labels'])

        # Info card
        st.markdown(f"""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#e8f1ff,#dfeaff);
                        border:1px solid #d2e4ff;margin-bottom:14px;">
                <h4 style="color:#2c3e50;margin:0;">📦 Enhanced Feature Matrix</h4>
                <p style="color:#1b3c70;font-size:18px;margin:4px 0;">
                    {feats.shape[0]} samples × {feats.shape[1]} features
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Styled feature stats table
        stats = pd.DataFrame(feats).describe().T
        st.dataframe(
            stats.head(20).style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

        # Correlation heatmap (UPDATED — blue → red palette)
        subset = pd.DataFrame(feats[:, :min(30, feats.shape[1])])
        fig, ax = plt.subplots(figsize=(8,6))

        if sns is not None:
            sns.heatmap(
                subset.corr(),
                cmap="coolwarm",    
                ax=ax,
                center=0,
                annot=False,
                cbar=True
            )
        else:
            ax.imshow(subset.corr(), cmap="coolwarm")

        ax.set_title(
            'Feature Correlation (First 30 Features)', 
            color="#1A1A1A"   
        )

        st.pyplot(fig)

    else:
        st.info('⚠️ enhanced_features.pkl not found in models/. Run feature_extrac.py first.')

# ---------- FEATURE EXTRACTION  ----------
elif page == 'Feature Extraction':

    # Header card
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px;
                    background-color: #f7f9fc; border: 1px solid #e6e9ef; margin-bottom: 20px;">
            <h2 style="color:#2c3e50; margin: 0;">⚙️ Feature Extraction</h2>
            <p style="font-size:16px; color:#4e5d6c; margin-top:8px;">
                Extract handcrafted features (PRNU, FFT, LBP) from images or entire datasets.
                All visuals follow the same soft–gradient UI used across the app.
            </p>
        </div>
    """, unsafe_allow_html=True)

    mode = st.selectbox('Choose Extraction Mode', [
        'Upload Images (quick)',
        'Extract from Dataset Folder',
        'Baseline Feature Explorer (recursive)'
    ])

    # -----------------------------------------------------------
    # 1. Upload Images (Quick Mode — CNN Style Extraction)
    # -----------------------------------------------------------
    if mode == 'Upload Images (quick)':
        st.markdown("""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#e8f1ff,#dfeaff);
                        border:1px solid #d2e4ff; margin-bottom:14px;">
                <h4 style="color:#2c3e50;margin:0;">📤 Upload Images for Feature Extraction</h4>
                <p style="color:#4e5d6c; margin:4px 0 0;">
                    Upload any images to extract PRNU, FFT, and LBP feature vectors instantly.
                </p>
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            'Upload one or more images',
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            accept_multiple_files=True
        )

        if uploaded:
            rows = []

            for u in uploaded:

                # Card for each uploaded image
                st.markdown(f"""
                    <div style="padding:12px;border-radius:12px;
                                background:#ffffff;
                                border:1px solid #e6e9ef;
                                margin-top:15px;">
                        <h4 style="color:#2c3e50;margin:0;">🖼️ Processing: {u.name}</h4>
                    </div>
                """, unsafe_allow_html=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(u.name).suffix) as tmp:
                    tmp.write(u.read())
                    tmp_path = tmp.name

                try:
                    res = preprocess_to_residual(tmp_path)

                    # residual preview styling
                    res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)
                    res_enhanced = np.clip(res_norm * 2.2, 0, 1)
                    res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[..., None], 3, axis=2)

                    st.image(
                        res_rgb,
                        caption="Enhanced Residual Preview",
                        use_container_width=True
                    )

                    # color-themed divider
                    st.markdown(
                        "<hr style='border: 1px solid #d9eaff; margin-top:10px; margin-bottom:20px;'>",
                        unsafe_allow_html=True
                    )

                    # Compute handcrafted features
                    fp = load_pickle_safe(MODELS_DIR / 'scanner_fingerprints.pkl')
                    if fp:
                        fp_keys = np.load(MODELS_DIR / 'fp_keys.npy', allow_pickle=True).tolist()
                        v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                    else:
                        v_corr = []

                    v_fft = fft_radial_energy(res, K=6)
                    v_lbp = lbp_hist_safe(res, P=8, R=1.0)

                    feat_vec = v_corr + v_fft + v_lbp

                    rows.append({
                        'filename': u.name,
                        'corr_len': len(v_corr),
                        'fft_len': len(v_fft),
                        'lbp_len': len(v_lbp),
                        'feature_vector': feat_vec
                    })

                except Exception as e:
                    st.error(f"❌ Error extracting {u.name}: {e}")

            if rows:
                df = pd.DataFrame(rows)

                # Styled table
                st.markdown("<h4 style='color:#2c3e50;'>📊 Extracted Feature Summary</h4>", unsafe_allow_html=True)
                st.dataframe(
                    df[['filename','corr_len','fft_len','lbp_len']]
                        .style.set_properties(**{
                            "background-color": "#ffffff",
                            "color": "#2c3e50",
                            "border": "1px solid #eaeaea",
                            "padding": "6px"
                        })
                        .set_table_styles([{
                            "selector": "th",
                            "props": [("background-color", "#e8f1ff"),
                                      ("color", "#2c3e50"),
                                      ("border", "1px solid #d9eaff")]
                        }]),
                    use_container_width=True
                )

                # Download button
                df2 = df.copy()
                df2['feature_vector'] = df2['feature_vector'].apply(lambda x: ','.join(map(str, x)))

                st.download_button(
                    "📥 Download Features as CSV",
                    data=df2.to_csv(index=False).encode(),
                    file_name='uploaded_features.csv',
                    mime='text/csv'
                )

    # -----------------------------------------------------------
    # 2. Extract From Entire Dataset Folder
    # -----------------------------------------------------------
    elif mode == 'Extract from Dataset Folder':

        st.markdown("""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#dff0ff,#c6e9ff);
                        border:1px solid #cfe8ff; margin-bottom:14px;">
                <h4 style="color:#2c3e50;margin:0;">📂 Dataset-Wide Extraction</h4>
                <p style="color:#4e5d6c;margin-top:4px;">
                    Automatically extracts features for ALL images in a folder.
                </p>
            </div>
        """, unsafe_allow_html=True)

        dataset_base = st.text_input('Dataset path:', value=str(DATA_DIR))

        if st.button('🚀 Start Extraction'):
            dataset_path = Path(dataset_base)

            if not dataset_path.exists():
                st.error(f'❌ Path not found: {dataset_path}')
            else:
                all_images = [p for p in dataset_path.rglob('*') if p.suffix.lower() in SUPPORTED_EXT]
                total_imgs = len(all_images)

                if total_imgs == 0:
                    st.warning("No images found.")
                else:
                    rows = []
                    st.info(f"Found **{total_imgs}** images. Extracting features...")
                    progress = st.progress(0)

                    sample_images = np.random.choice(all_images, size=min(5, total_imgs), replace=False)

                    for i, img_path in enumerate(all_images, 1):

                        try:
                            res = preprocess_to_residual(str(img_path))

                            # show a few samples
                            if img_path in sample_images:
                                res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)
                                res_enhanced = np.clip(res_norm * 2.2, 0, 1)
                                res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[..., None], 3, axis=2)
                                st.image(
                                    res_rgb,
                                    caption=f"Residual Sample — {img_path.name}",
                                    use_container_width=True
                                )

                            fp = load_pickle_safe(MODELS_DIR / 'scanner_fingerprints.pkl')
                            if fp:
                                fp_keys = np.load(MODELS_DIR / 'fp_keys.npy', allow_pickle=True).tolist()
                                v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                            else:
                                v_corr = []

                            v_fft = fft_radial_energy(res, K=6)
                            v_lbp = lbp_hist_safe(res, P=8, R=1.0)

                            feat_vec = v_corr + v_fft + v_lbp

                            rows.append({
                                'filepath': str(img_path),
                                'scanner': img_path.parent.name,
                                'feature_vector': feat_vec
                            })

                        except Exception as e:
                            st.warning(f"⚠️ Skipped {img_path.name}: {e}")

                        progress.progress(i / total_imgs)

                    if rows:
                        df = pd.DataFrame(rows)

                        # Polished success banner
                        st.markdown("""
                            <div style="padding:14px;border-radius:12px;
                                        background:linear-gradient(135deg,#e8ffe8,#d0f5d0);
                                        border:1px solid #b6e6b6; margin-top:14px;">
                                <h4 style="color:#2c3e50;margin:0;">✅ Dataset Feature Extraction Completed</h4>
                            </div>
                        """, unsafe_allow_html=True)

                        st.dataframe(df.head(12), use_container_width=True)

                        df2 = df.copy()
                        df2['feature_vector'] = df2['feature_vector'].apply(lambda x: ','.join(map(str, x)))
                        st.download_button(
                            "📥 Download Dataset Features (CSV)",
                            data=df2.to_csv(index=False).encode(),
                            file_name='dataset_extracted_features.csv',
                            mime='text/csv'
                        )

    # -----------------------------------------------------------
    # 3. Baseline Recursive Feature Explorer
    # -----------------------------------------------------------
    elif mode == 'Baseline Feature Explorer (recursive)':

        st.markdown("""
            <div style="padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#f5fbff,#e2f1ff);
                        border:1px solid #d9eaff; margin-bottom:14px;">
                <h4 style="color:#2c3e50;margin:0;">📁 Baseline Recursive Feature Explorer</h4>
                <p style="color:#4e5d6c;margin-top:4px;">
                    Recursively scans directories, extracts features per class, and computes class statistics.
                </p>
            </div>
        """, unsafe_allow_html=True)

        dataset_root = st.text_input("Enter dataset root path:")

        if dataset_root and os.path.isdir(dataset_root):

            st.info("🔍 Scanning dataset recursively...")

            records = []
            class_dirs = set()

            for dirpath, _, filenames in os.walk(dataset_root):
                rel = os.path.relpath(dirpath, dataset_root)
                if rel == '.':
                    continue

                class_name = rel.split(os.sep)[0]
                class_dirs.add(class_name)

                img_files = [f for f in filenames if f.lower().endswith(SUPPORTED_EXT)]

                for fname in img_files:
                    img_path = os.path.join(dirpath, fname)
                    rec = extract_features(img_path, class_name)
                    records.append(rec)

            if records:
                df = pd.DataFrame(records)

                st.markdown(f"""
                    <div style="padding:14px;border-radius:12px;
                                background:linear-gradient(135deg,#e8f1ff,#bcd9ff);
                                border:1px solid #d2e4ff; margin-bottom:14px;">
                        <h4 style="color:#2c3e50;margin:0;">📦 Extracted {len(df)} feature records</h4>
                        <p style="color:#4e5d6c;margin:4px 0 0;">
                            Detected classes: {list(class_dirs)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.dataframe(df.head(20), use_container_width=True)

                save_path = os.path.join(dataset_root, 'metadata_features.csv')
                df.to_csv(save_path, index=False)

                st.success(f"💾 Saved metadata to {save_path}")

                if 'class' in df.columns:
                    st.subheader("📈 Class Distribution")
                    st.bar_chart(df['class'].value_counts())

            else:
                st.warning("⚠️ No supported images found in the directory.")

        elif dataset_root:
            st.error("❌ Invalid dataset path.")

# ---------- MODAL TRAINING & EVALUATION (Soft Blue Gradient UI) ----------
elif page == 'Modal Training & Evaluation':

    # Main header card
    st.markdown("""
        <div style="padding:20px; border-radius:12px;
                    background:linear-gradient(135deg, #f7faff, #e8f3ff);
                    border:1px solid #dbe7ff; margin-bottom:20px;">
            <h2 style="color:#2c3e50; margin:0;">🧠 Modal Training & Evaluation</h2>
            <p style="color:#4e5d6c; font-size:16px; margin-top:8px;">
                Train and evaluate both <b>Classical ML Models</b> and the advanced <b>Hybrid CNN</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "🌲 Baseline Models", 
        "🧩 Hybrid CNN Model"
    ])

    # --------------------------------------------------------------------
    # 🌲 TAB 1: BASELINE MODELS (Soft Blue UI)
    # --------------------------------------------------------------------
    with tab1:
        st.markdown("""
            <h3 style='color:#2c3e50;'>🌲 Baseline Models — RandomForest & SVM</h3>
            <p style='color:#4e5d6c;'>
                These classical models use handcrafted features (entropy, edges, histograms, PRNU-like stats).
            </p>
        """, unsafe_allow_html=True)

        colA, colB = st.columns([1.2, 1])

        # TRAIN MODELS BUTTON CARD
        with colA:
            st.markdown("""
                <div style="padding:14px; border-radius:12px;
                    background:linear-gradient(135deg,#e8f1ff,#bcd9ff);
                    border:1px solid #cddfff;">
                    <h4 style="color:#2c3e50;margin:0;">🚀 Train Baseline Models</h4>
                    <p style="color:#1b3c70; margin:6px 0 12px;">
                        Trains RandomForest + SVM using <b>official.csv</b>.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            if not os.path.exists(CSV_PATH):
                st.warning("⚠️ `official.csv` not found.")
            else:
                if st.button("🚀 Start Training", use_container_width=True):
                    with st.spinner("Training Random Forest & SVM..."):
                        start_time = time.time()
                        try:
                            train_baseline_models()
                            elapsed = time.time() - start_time
                            st.success(f"✅ Completed in {elapsed:.2f}s | Models saved to /models/")
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")

        # EVALUATION BUTTONS CARD
        with colB:
            st.markdown("""
                <div style="padding:14px; border-radius:12px;
                    background:linear-gradient(135deg,#dff0ff,#c6e9ff);
                    border:1px solid #cfe8ff;">
                    <h4 style="color:#2c3e50;margin:0;">🧾 Evaluate Trained Models</h4>
                    <p style="color:#1b3c70;margin:6px 0 12px;">Evaluate saved baseline models.</p>
                </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("📊 Random Forest"):
                    evaluate_baseline_model(MODELS_DIR / "random_forest.pkl", "Random Forest")
            with c2:
                if st.button("📈 SVM"):
                    evaluate_baseline_model(MODELS_DIR / "svm.pkl", "SVM")

        st.markdown("<hr>", unsafe_allow_html=True)

        with st.expander("💡 Baseline Model Info"):
            st.write("""
            - 🌲 **Random Forest:** 300-tree ensemble.
            - ⚡ **SVM (RBF):** High-dimensional nonlinear classifier.
            - 📦 **Scaler:** Stored as `models/scaler.pkl`.
            """)

    # --------------------------------------------------------------------
    # 🧩 TAB 2: HYBRID CNN MODEL (Soft Blue Premium UI)
    # --------------------------------------------------------------------
    with tab2:

        # Section header
        st.markdown("""
            <h3 style='color:#2c3e50;'>🧩 Hybrid CNN — Deep Feature Fusion</h3>
            <p style='color:#4e5d6c;'>
                CNN-based residual learning combined with handcrafted FFT, LBP, and PRNU correlations.
            </p>
        """, unsafe_allow_html=True)

        # ---------------- SELECTORS ----------------
        st.markdown("<h4 style='color:#2c3e50;'>📦 Select Model & Feature Artifacts</h4>", unsafe_allow_html=True)
        colM, colF = st.columns(2)

        with colM:
            model_files = {
                "Hybrid CNN (Final)": MODELS_DIR / "scanner_hybrid_final.keras",
                "Hybrid CNN (Checkpoint)": MODELS_DIR / "scanner_hybrid.keras"
            }
            selected_model_label = st.selectbox("📦 CNN Model File", list(model_files.keys()))
            model_path = model_files[selected_model_label]

        with colF:
            feature_files = {
                "Enhanced Features": MODELS_DIR / "enhanced_features.pkl",
                "Baseline Features": MODELS_DIR / "features.pkl"
            }
            selected_feat_label = st.selectbox("🧮 Feature Data", list(feature_files.keys()))
            feature_path = feature_files[selected_feat_label]

        # Artifacts
        encoder_path = MODELS_DIR / "hybrid_label_encoder.pkl"
        scaler_path = MODELS_DIR / "hybrid_feat_scaler.pkl"
        hist_path = MODELS_DIR / "hybrid_training_history.pkl"

        # ---------------- Load Model ----------------
        if not model_path.exists():
            st.error(f"❌ Model not found: {model_path.name}")
        else:
            st.success(f"✅ Loaded: {model_path.name}")

            if TF_AVAILABLE:

                @st.cache_resource(show_spinner=False)
                def load_tf_model_cached(path):
                    return tf.keras.models.load_model(path, compile=False)

                with st.spinner("Loading TensorFlow model..."):
                    model = load_tf_model_cached(str(model_path))

                # Info card
                st.markdown(f"""
                    <div style="padding:14px; border-radius:12px;
                        background:linear-gradient(135deg,#f5fbff,#e2f1ff);
                        border:1px solid #d9eaff;">
                        <h4 style="color:#2c3e50;margin:0;">🧬 Model Architecture</h4>
                        <p style="color:#1b3c70;margin:4px 0;">Layers: <b>{len(model.layers)}</b></p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("⚠️ TensorFlow not available.")
                st.stop()

            # ---------------- Training History ----------------
            if hist_path.exists():
                with st.expander("📉 Training History"):
                    hist = load_pickle_safe(hist_path)
                    if hist:
                        fig, ax = plt.subplots(1, 2, figsize=(10,4))
                        ax[0].plot(hist.get("accuracy", []), label="Train")
                        ax[0].plot(hist.get("val_accuracy", []), label="Val")
                        ax[0].set_title("Accuracy")
                        ax[0].legend()

                        ax[1].plot(hist.get("loss", []), label="Train")
                        ax[1].plot(hist.get("val_loss", []), label="Val")
                        ax[1].set_title("Loss")
                        ax[1].legend()

                        st.pyplot(fig)

            # ---------------- Evaluation ----------------
            st.markdown("<h4 style='color:#2c3e50;'>🧠 Run Evaluation</h4>", unsafe_allow_html=True)

            if feature_path.exists():
                ef = load_pickle_safe(feature_path)
                feats = np.array(ef["features"], dtype=np.float32)
                labels = np.array(ef["labels"])

                st.markdown(f"""
                    <div style="padding:14px; border-radius:12px;
                        background:linear-gradient(135deg,#e8f1ff,#dfeaff);
                        border:1px solid #d2e4ff; margin-bottom:12px;">
                        <h4 style="color:#2c3e50;margin:0;">📦 Feature Matrix Loaded</h4>
                        <p style="color:#1b3c70;margin:4px 0;">
                            {feats.shape[0]} samples × {feats.shape[1]} features
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Slider UI
                min_val, max_val = 1, feats.shape[0]
                default_val = min(1000, max_val)
                max_samples = st.slider("Limit evaluation to N samples", min_val, max_val, default_val, step=50)

                idx = np.random.choice(feats.shape[0], max_samples, replace=False)
                feats_sub, labels_sub = feats[idx], labels[idx]

                scaler = load_pickle_safe(scaler_path)
                encoder = load_pickle_safe(encoder_path)

                if scaler is not None and getattr(scaler, "n_features_in_", feats_sub.shape[1]) == feats_sub.shape[1]:
                    feats_sub = scaler.transform(feats_sub)
                else:
                    st.caption("⚠️ Skipping scaling due to mismatch.")

                expected_dim = model.inputs[1].shape[1]
                expected_dim = int(expected_dim)

                if feats_sub.shape[1] != expected_dim:
                    feats_sub = (
                        feats_sub[:, :expected_dim]
                        if feats_sub.shape[1] > expected_dim
                        else np.pad(feats_sub, ((0,0),(0, expected_dim - feats_sub.shape[1])), mode="constant")
                    )

                dummy_img = np.zeros((feats_sub.shape[0], 256, 256, 1), dtype=np.float32)

                if st.button("▶️ Run Evaluation", use_container_width=True):
                    st.info("⏳ Running inference...")
                    progress = st.progress(0)

                    preds_list = []
                    batch_size = 256

                    for i in range(0, len(feats_sub), batch_size):
                        batch_feats = feats_sub[i:i+batch_size]
                        batch_imgs = dummy_img[i:i+batch_size]
                        preds_list.append(model.predict([batch_imgs, batch_feats], verbose=0))
                        progress.progress(min(1.0, (i + batch_size) / len(feats_sub)))

                    preds = np.vstack(preds_list)
                    y_pred = np.argmax(preds, axis=1)

                    if encoder is not None:
                        y_true = encoder.transform(labels_sub)
                        report = classification_report(
                            y_true, y_pred, target_names=encoder.classes_, output_dict=True
                        )

                        # Results table styled
                        st.success("✅ Evaluation Complete!")
                        st.dataframe(
                            pd.DataFrame(report).T.style.background_gradient(cmap="Blues"),
                            use_container_width=True
                        )

                        # Confusion Matrix
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(max(8, len(encoder.classes_)*0.5), 6))
                        try:
                            from sklearn.metrics import ConfusionMatrixDisplay
                            ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(ax=ax, cmap="Blues")
                        except:
                            ax.imshow(cm, cmap="Blues")
                        ax.set_title("Predicted vs True Scanners", color="#2c3e50")
                        st.pyplot(fig)

                        # Class-wise accuracy
                        with st.expander("📈 Class-wise Recall / Accuracy"):
                            accs = [report[c]["recall"] for c in encoder.classes_ if c in report]
                            fig, ax = plt.subplots(figsize=(10,4))
                            ax.bar(encoder.classes_, accs, color="#8fb4ff")
                            ax.set_ylim(0,1)
                            ax.set_ylabel("Recall")
                            plt.xticks(rotation=45, ha="right")
                            st.pyplot(fig)

                    else:
                        st.warning("Label encoder missing — cannot compute metrics.")

            else:
                st.warning(f"⚠️ Feature file missing: {feature_path}")

# ---------- LIVE PREDICTION (Soft Blue Gradient UI) ----------
elif page == 'Live Prediction':

    # Header card
    st.markdown("""
        <div style="padding:20px; border-radius:12px;
                    background:linear-gradient(135deg,#f7faff,#e8f2ff);
                    border:1px solid #dbe7ff; margin-bottom:20px;">
            <h2 style="color:#2c3e50; margin:0;">🔮 Live Prediction</h2>
            <p style="color:#4e5d6c; font-size:16px; margin-top:8px;">
                Upload an image and let the model identify the <b>source scanner</b>.
                Supports both Baseline and Hybrid CNN models.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌲 Baseline Prediction", "🧩 Hybrid CNN Prediction"])

    # ==========================================================
    # 🌲 TAB 1 — BASELINE PREDICTION (Soft Blue)
    # ==========================================================
    with tab1:

        st.markdown("""
            <h3 style='color:#2c3e50;'>🌲 Baseline Prediction — RandomForest & SVM</h3>
            <p style='color:#4e5d6c;'>Predict scanner using handcrafted features only.</p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📄 Upload a Document Image",
            type=list(SUPPORTED_EXT),
            key="baseline_upload"
        )

        col1, col2 = st.columns([1,1])
        with col1:
            model_choice = st.selectbox(
                "⚙️ Select Model",
                ["rf", "svm"],
                key="baseline_model_choice"
            )
        with col2:
            st.info("Train models from the 'Modal Training & Evaluation' page.")

        # ---------------- PROCESS ----------------
        if uploaded_file is not None:

            st.markdown("<h4 style='color:#2c3e50;'>📸 Uploaded Image</h4>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)

            if st.button("🚀 Predict (Baseline)", use_container_width=True, key="baseline_predict"):
                tmp_path = MODELS_DIR / f"temp_baseline{Path(uploaded_file.name).suffix}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.read())

                try:
                    with st.spinner("Running baseline prediction..."):
                        start = time.time()
                        pred, prob, classes = predict_baseline_scanner(tmp_path, model_choice)
                        elapsed = time.time() - start

                    # -------- Prediction Card --------
                    st.markdown(f"""
                        <div style="padding:16px; border-radius:12px;
                            background:linear-gradient(135deg,#e7f0ff,#d1e4ff);
                            border:1px solid #c7dcff;margin-top:10px;">
                            <h3 style="color:#1b3462;margin:0;">🎯 Prediction Result</h3>
                            <p style="color:#2c3e50; margin:6px 0 0;">
                                <b>Scanner:</b> {pred}<br>
                                <b>Time:</b> {elapsed:.2f}s
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    # -------- Probability Chart --------
                    st.markdown("<h4 style='color:#2c3e50;'>🔢 Class Probabilities</h4>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.bar(classes, prob, color="#8bb6ff")
                    ax.set_ylabel("Probability")
                    ax.set_ylim(0,1)
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Baseline prediction failed: {e}")

        # Info expander
        with st.expander("ℹ️ About Baseline Prediction"):
            st.markdown("""
            - Uses **handcrafted features**  
            - Models: RF, SVM  
            - Works well on simple datasets  
            """)

    # ==========================================================
    # 🧩 TAB 2 — HYBRID CNN PREDICTION (Soft Blue Premium UI)
    # ==========================================================
    with tab2:

        st.markdown("""
            <h3 style='color:#2c3e50;'>🧩 Hybrid CNN — Deep Fusion Prediction</h3>
            <p style='color:#4e5d6c;'>Combines residual CNN features + handcrafted features.</p>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "📄 Upload a Document Image",
            type=list(SUPPORTED_EXT),
            key="cnn_upload"
        )

        model_files = {
            "Hybrid CNN (Final)": MODELS_DIR / "scanner_hybrid_final.keras",
            "Hybrid CNN (Checkpoint)": MODELS_DIR / "scanner_hybrid.keras"
        }
        chosen_model = st.selectbox("🧠 Select Model File", list(model_files.keys()))
        model_path = model_files[chosen_model]

        st.info("Ensure required artifacts: fingerprints.pkl, scaler.pkl, label_encoder.pkl")

        # ---------------- PROCESS ----------------
        if uploaded:

            st.markdown("<h4 style='color:#2c3e50;'>📸 Uploaded Image</h4>", unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)

            if st.button("🚀 Predict (Hybrid CNN)", use_container_width=True, key="cnn_predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
                tmp.write(uploaded.getvalue())
                tmp.close()

                try:
                    # ------------------- PREPROCESS -------------------
                    with st.spinner("🔄 Extracting residual & handcrafted features..."):

                        res = preprocess_to_residual(tmp.name)
                        x_img = np.expand_dims(res, axis=(0, -1)).astype(np.float32)

                        fp = load_pickle_safe(MODELS_DIR / "scanner_fingerprints.pkl")
                        if fp:
                            fp_keys = np.load(MODELS_DIR / 'fp_keys.npy', allow_pickle=True)
                            v_corr = [corr2d(res, fp[k]) for k in fp_keys]
                        else:
                            v_corr = []

                        v_fft = fft_radial_energy(res, K=6)
                        v_lbp = lbp_hist_safe(res, P=8, R=1.0)

                        feat_vec = np.array([v_corr + v_fft + v_lbp], dtype=np.float32)

                        scaler = load_pickle_safe(MODELS_DIR / "hybrid_feat_scaler.pkl")
                        if scaler is not None:
                            try:
                                feat_vec = scaler.transform(feat_vec)
                            except:
                                st.warning("⚠️ Scaler mismatch — using raw features.")

                    # ------------------- LOAD MODEL -------------------
                    if not model_path.exists():
                        st.error(f"❌ Model missing: {model_path}")
                    else:
                        with st.spinner("🧠 Loading CNN model..."):
                            model = tf.keras.models.load_model(str(model_path))

                        # ------------------- PREDICT -------------------
                        st.info("⏳ Predicting...")
                        progress = st.progress(0)

                        prob = model.predict([x_img, feat_vec], verbose=0)
                        progress.progress(1.0)

                        idx = int(np.argmax(prob, axis=1))
                        encoder = load_pickle_safe(MODELS_DIR / "hybrid_label_encoder.pkl")
                        label = encoder.classes_[idx] if encoder else str(idx)
                        conf = float(prob[0, idx] * 100)

                        # -------- Prediction Result Card --------
                        st.markdown(f"""
                            <div style="padding:18px; border-radius:12px;
                                background:linear-gradient(135deg,#e4efff,#d1e4ff);
                                border:1px solid #c9dcff;margin-top:15px;">
                                <h3 style="color:#1b3462;margin:0;">🎯 Hybrid CNN Prediction</h3>
                                <p style="color:#2c3e50;margin:6px 0;">
                                    <b>Scanner:</b> {label}<br>
                                    <b>Confidence:</b> {conf:.2f}%
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                        # -------- Probability Chart --------
                        st.markdown("<h4 style='color:#2c3e50;'>🔢 Class Probabilities</h4>", unsafe_allow_html=True)
                        classes = encoder.classes_ if encoder else np.arange(prob.shape[1])
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.bar(classes, prob[0], color="#8fb4ff")
                        ax.set_ylim(0,1)
                        ax.set_ylabel("Probability", color="#2c3e50")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)

                        # -------- Residual Visualization --------
                        with st.expander("📸 Enhanced Residual Preview"):
                            res_norm = (res - np.min(res)) / (np.max(res) - np.min(res) + 1e-9)
                            res_enhanced = np.clip(res_norm * 1.5, 0, 1)
                            res_rgb = np.repeat((res_enhanced * 255).astype(np.uint8)[...,None], 3, axis=2)

                            st.markdown("""
                                <div style="padding:10px; border-radius:10px;
                                    background:#eef4ff; border:1px solid #cddcff;">
                                    <h4 style="color:#2c3e50;">Residual Visualization</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            st.image(res_rgb, use_container_width=False)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

        # Info expander
        with st.expander("ℹ️ About Hybrid CNN Prediction"):
            st.markdown("""
            - CNN learns scanner noise patterns  
            - Handcrafted features add PRNU, FFT, LBP texture information  
            - More robust for complex & unseen data  
            """)


# ---------- FORGERY / TAMPER DETECTION (new) ----------
elif page == 'Forgery / Tamper Detection':
    st.header("🔎 Forgery / Tamper Detection")
    st.write("Two tools in one: scanner identification (hybrid CNN + handcrafted) and image tamper detection (image-level + patch fallback).")

    import json, math
    from pathlib import Path as _Path

    TAMPER_DIR = MODELS_DIR / "tampered module"

    SCN_MODEL_PATH  = str(TAMPER_DIR / "scanner_hybrid.keras")
    SCN_LE_PATH     = str(TAMPER_DIR / "hybrid_label_encoder.pkl")
    SCN_SCALER_PATH = str(TAMPER_DIR / "hybrid_feat_scaler.pkl")
    SCN_FP_PATH     = str(TAMPER_DIR / "scanner_fingerprints.pkl")
    SCN_FP_KEYS     = str(TAMPER_DIR / "fp_keys.npy")

    IMG_SCALER_PATH = str(TAMPER_DIR / "image_scaler.pkl")
    IMG_CLF_PATH    = str(TAMPER_DIR / "image_svm_sig.pkl")
    IMG_THR_JSON    = str(TAMPER_DIR / "image_thresholds.json")

    TP_SCALER_PATH  = str(TAMPER_DIR / "patch_scaler.pkl")
    TP_CLF_PATH     = str(TAMPER_DIR / "patch_svm_sig_calibrated.pkl")
    TP_THR_JSON     = str(TAMPER_DIR / "thresholds_patch.json")

    PATCH = 128
    STRIDE = 64
    MAX_PATCHES = 16

    @st.cache_resource
    def _load_tf_model(path):
        try: return tf.keras.models.load_model(path, compile=False)
        except: return None

    @st.cache_resource
    def _load_pickle(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except: return None

    @st.cache_resource
    def _load_npy_list(path):
        try: return np.load(path, allow_pickle=True).tolist()
        except: return None

    @st.cache_resource
    def _load_json(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return None

    scanner_model = _load_tf_model(SCN_MODEL_PATH)
    scanner_le    = _load_pickle(SCN_LE_PATH)
    scanner_scaler= _load_pickle(SCN_SCALER_PATH)
    scanner_fps   = _load_pickle(SCN_FP_PATH)
    fp_keys       = _load_npy_list(SCN_FP_KEYS)

    IMG_AVAILABLE = True
    image_scaler = _load_pickle(IMG_SCALER_PATH)
    image_clf    = _load_pickle(IMG_CLF_PATH)
    THR_IMG      = _load_json(IMG_THR_JSON)
    if image_scaler is None or image_clf is None:
        IMG_AVAILABLE = False

    PATCH_AVAILABLE = True
    patch_scaler = _load_pickle(TP_SCALER_PATH)
    patch_clf    = _load_pickle(TP_CLF_PATH)
    THR_PATCH    = _load_json(TP_THR_JSON)
    if patch_scaler is None or patch_clf is None:
        PATCH_AVAILABLE = False

    SCANNER_READY = (
        scanner_model is not None and
        scanner_le is not None and
        scanner_scaler is not None and
        scanner_fps is not None and
        fp_keys is not None
    )

    # --------------------------------------------------
    # helper functions 
    # --------------------------------------------------
    def _make_scanner_feats_from_res(res):
        if fp_keys is None or scanner_fps is None:
            raise RuntimeError("Scanner fingerprints missing.")
        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
        v_fft  = fft_radial_energy(res, 6)
        v_lbp  = lbp_hist_safe(res, 8, 1.0)
        feat = np.array(v_corr + v_fft + v_lbp, np.float32).reshape(1, -1)
        return scanner_scaler.transform(feat)

    def _predict_scanner_from_path(path):
        res = preprocess_to_residual(path)
        x_img = np.expand_dims(res, axis=(0, -1)).astype(np.float32)
        x_feat = _make_scanner_feats_from_res(res)
        prob = scanner_model.predict([x_img, x_feat], verbose=0).ravel()
        idx = int(np.argmax(prob))
        label = scanner_le.classes_[idx]
        return label, prob[idx]*100, prob


    # ------------------------------------------------------
    # UI TABS
    # ------------------------------------------------------

    tab_scanner, tab_tamper = st.tabs(["🔎 Scanner Identification", "⚠️ Tamper Analysis"])

    # ------------------------------------------------------
    # TAB 1 - Scanner Identification
    # ------------------------------------------------------
    with tab_scanner:
        st.subheader("🔎 Scanner Identification (27-D hybrid)")

        if not SCANNER_READY:
            st.warning("Scanner artifacts missing in 'tampered module'.")

        upload_sc = st.file_uploader(
            "Upload image for scanner identification",
            type=list(SUPPORTED_EXT),
            key="tamper_scan_upload"
        )

        if upload_sc is not None:
            st.image(upload_sc, caption="Uploaded Image", use_container_width=True)

            if st.button("Predict Scanner", key="btn_predict_scanner"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload_sc.name).suffix)
                tmp.write(upload_sc.getvalue())
                tmp.close()

                try:
                    if not SCANNER_READY:
                        st.error("Scanner model not available.")
                        st.stop()

                    label, conf, prob = _predict_scanner_from_path(tmp.name)
                    st.success(f"Predicted: **{label}**  •  {conf:.2f}%")

                    with st.expander("Raw Probabilities"):
                        st.json({scanner_le.classes_[i]: float(prob[i]) for i in range(len(prob))})

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    os.remove(tmp.name)

    # ------------------------------------------------------
    # TAB 2 - Tamper Analysis
    # ------------------------------------------------------
    with tab_tamper:
        st.subheader("⚠️ Tamper Analysis (Image-level + Patch fallback)")

        if not IMG_AVAILABLE and not PATCH_AVAILABLE:
            st.warning("No tamper detection artifacts found.")
            st.stop()

        upload_t = st.file_uploader(
            "Upload image for tamper analysis",
            type=list(SUPPORTED_EXT),
            key="tamper_analysis_upload"
        )

        if upload_t is not None:
            st.image(upload_t, caption="Uploaded image", use_container_width=True)

            if st.button("Run Tamper Check", key="btn_run_tamper"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload_t.name).suffix)
                tmp.write(upload_t.getvalue())
                tmp.close()

                try:
                    if IMG_AVAILABLE:
                        tinfo = _infer_tamper_image(tmp.name)
                    elif PATCH_AVAILABLE:
                        tinfo = _infer_tamper_single_patch(tmp.name)

                    st.metric("Tamper label", tinfo["tamper_label"])
                    st.write(f"Probability: {tinfo['prob_tampered']:.3f}")
                    st.write(f"Threshold: {tinfo['threshold']:.3f}")
                    st.write(f"Confidence: {tinfo['confidence']:.2f}%")

                    with st.expander("Full Debug Info"):
                        st.json(tinfo)

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    os.remove(tmp.name)

# ---------- ABOUT ----------
elif page == 'About':

    st.markdown("""
        <style>
            .about-card {
                background: linear-gradient(135deg, #e8f0fe 0%, #ffffff 100%);
                padding: 25px 28px;   /* reduced padding */
                border-radius: 18px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border: 1px solid #d0ddf5;
            }
            .about-header {
                font-size: 32px;          /* slight reduction */
                font-weight: 800;
                color: #1a3d7c;
                text-align: center;
                margin-bottom: 4px;       /* reduced bottom spacing */
                margin-top: 0px;          /* remove top gap */
                padding: 4px 0px;         /* tighten header box spacing */
            }
            .section-title {
                font-size: 22px;
                font-weight: 600;
                color: #1a3d7c;
                margin-top: 25px;
            }
            .footer-box {
                margin-top: 35px;
                padding: 18px;
                border-radius: 12px;
                background: #f0f5ff;
                border-left: 5px solid #3b63d1;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown("""
        <div class="about-card">
            <div class="about-header">ℹ️ About — AI TraceFinder</div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    **AI TraceFinder** is a unified forensic machine learning suite for  
    **Scanner Source Identification** and **Forgery / Tamper Detection**.

    It analyzes **scanner PRNU noise fingerprints**, **frequency signatures**,  
    and **micro-texture statistics** to detect the origin and authenticity of a document.
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🧠 Core Forensic Concept</div>', unsafe_allow_html=True)
    st.markdown("""
    Every scanner leaves unique hidden patterns during image acquisition.

    AI TraceFinder captures those patterns using:
    - 🔍 **PRNU Correlation** (sensor noise fingerprint similarity)  
    - 📡 **FFT Radial Frequency Energy**  
    - 🌐 **LBP Micro-Texture Histograms**  
    These features are fused with a **Hybrid CNN** that learns signature-level noise residuals.
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">⚙️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    | Model | Description |
    |-------|-------------|
    | 🧩 **Hybrid CNN Model** | Learns scanner-specific residual patterns + fused handcrafted features |
    | 🌲 **Random Forest** | Baseline model for handcrafted features |
    | 🔹 **SVM (RBF)** | Traditional classifier for quick evaluation |
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🧩 Technologies Used</div>', unsafe_allow_html=True)
    st.markdown("""
    - 🎛 **Streamlit** – UI & dashboard framework  
    - 🖼 **OpenCV, PyWavelets** – residual extraction & denoising  
    - 📊 **NumPy, Pandas** – dataset & feature handling  
    - 🤖 **TensorFlow / Keras** – Hybrid CNN training  
    - 📈 **scikit-learn** – classical ML models  
    - 🎨 **Matplotlib / Seaborn / Plotly** – visual analytics  
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">📦 System Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Residual Extraction** → Removes image content, isolates scanner noise  
    2. **Handcrafted Feature Generation** → PRNU, FFT, LBP, patch stats  
    3. **CNN Feature Learning** → Deep extraction from residual maps  
    4. **Feature Fusion** → CNN features + handcrafted features  
    5. **Classification** → Identify scanner / detect tampering  
    6. **Visualization** → Confusion matrices, correlation maps, model performance  
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🛠 Current Setup Requirements</div>', unsafe_allow_html=True)
    st.markdown("""
    - Store trained models in **/models/**  
    - Keep hybrid artifacts in **/models/tampered module/**  
    - Place datasets inside **/Datasets/** folder  
    - App auto-detects:
      - `.keras` models  
      - `.pkl` artifacts  
      - `.csv` metadata  
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🔒 Security & Forensic Value</div>', unsafe_allow_html=True)
    st.markdown("""
    AI TraceFinder is designed for digital forensics labs, research institutions,  
    document authentication systems, and law-enforcement use cases.

    Key capabilities:
    - Detect forged or tampered document regions  
    - Identify which scanner device produced a document  
    - Compare noise fingerprints  
    - Evaluate authenticity using patch-level analysis  
    """)

    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🚀 Vision & Future Extensions</div>', unsafe_allow_html=True)
    st.markdown("""
    - Multi-scanner fingerprint clustering  
    - GAN-based synthetic forgery detection  
    - Heatmap localization for tampered regions  
    - Cloud deployment & API endpoints  
    - Real-time batch processing for forensic labs  
    """)

    # ------------------------------------------------------------------
    st.markdown("""
        <div class="footer-box">
            <b>👨‍💻 Developed By:</b> Nandlal Yadav  
            <br><b>🌐 GitHub Repository:</b>  
            <a href="https://github.com/Nandlal1412/AI_TraceFinder.git" target="_blank">
                My GitHub Repo
            </a>
        </div>
    """, unsafe_allow_html=True)

    # ------------------ CLOSE CARD ------------------
    st.markdown("</div>", unsafe_allow_html=True)

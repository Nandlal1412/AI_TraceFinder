# baseline app

import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import joblib
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# === CONFIG ===
st.set_page_config(page_title="Forgery Detection Dashboard", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📁 Feature Explorer", "📊 Model Evaluation", "🧪 Predict Scanner", "🛠️ Train Models"])

CSV_PATH = "official.csv"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# === FEATURE EXTRACTION ===
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

# === MODEL TRAINING ===
def train_models():
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.pkl")

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "models/svm.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

# === MODEL EVALUATION ===
def evaluate_model(model_path, name, save_dir="results"):
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]
    scaler = joblib.load("models/scaler.pkl")
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

# === PREDICTION ===
def predict_scanner(img_path, model_choice="rf"):
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(f"models/{'random_forest.pkl' if model_choice == 'rf' else 'svm.pkl'}")

    pil_img = Image.open(img_path).convert("L")
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
        "width": w, "height": h, "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb, "mean_intensity": mean_intensity,
        "std_intensity": std_intensity, "skewness": skewness,
        "kurtosis": kurt, "entropy": ent, "edge_density": edge_density
    }])

    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob

# === PAGE LOGIC ===
if page == "📁 Feature Explorer":
    st.title("📁 Feature Extraction Viewer")
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

elif page == "📊 Model Evaluation":
    st.title("📊 Evaluate Trained Models")
    if st.button("Evaluate Random Forest"):
        evaluate_model("models/random_forest.pkl", "Random Forest")
    if st.button("Evaluate SVM"):
        evaluate_model("models/svm.pkl", "SVM")

elif page == "🧪 Predict Scanner":
    st.title("🧪 Predict Document Scanner")
    uploaded_file = st.file_uploader("Upload a document image", type=SUPPORTED_EXTENSIONS)
    model_choice = st.selectbox("Choose model", ["rf", "svm"])
    if uploaded_file:
        temp_path = "temp_image.tif"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        pred, prob = predict_scanner(temp_path, model_choice)
        st.success(f"Predicted Scanner: {pred}")
        st.write("Class Probabilities:")
        st.json({cls: round(prob[i], 3) for i, cls in enumerate(joblib.load(f"models/{'random_forest.pkl' if model_choice == 'rf' else 'svm.pkl'}").classes_)})

elif page == "🛠️ Train Models":
    st.title("🛠️ Train Baseline Models")
    st.markdown("Use this section to retrain your models using the latest `official.csv` dataset.")

    if not os.path.exists(CSV_PATH):
        st.error("❌ Dataset file 'official.csv' not found in project root.")
    else:
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models..."):
                try:
                    train_models()
                    st.success("✅ Models trained and saved successfully!")
                    st.info("Models saved to `models/` folder:\n- random_forest.pkl\n- svm.pkl\n- scaler.pkl")
                except Exception as e:
                    st.error(f"⚠️ Training failed: {e}")        
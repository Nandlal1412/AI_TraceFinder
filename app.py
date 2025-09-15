import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="Forgery Dataset Feature Extractor", layout="wide")
st.title("✍️ Forged Handwritten Document Database - Auto Class Detection & Feature Extraction")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

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
        file_size = os.path.getsize(image_path) / 1024  # KB
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
        if img_files:
            st.write(f"📁 Class '{class_name}' → {len(img_files)} images")

        for fname in img_files:
            img_path = os.path.join(dirpath, fname)
            rec = extract_features(img_path, class_name)
            records.append(rec)

    if records:
        st.success(f"✅ Detected {len(class_dirs)} classes: {list(class_dirs)}")
        df = pd.DataFrame(records)
        st.subheader("📊 Features Extracted (Preview)")
        st.dataframe(df.head(20))

        save_path = os.path.join(dataset_root, "metadata_features.csv")
        df.to_csv(save_path, index=False)
        st.success(f"💾 Features saved to {save_path}")

        if "class" in df.columns:
            st.subheader("📈 Class Distribution")
            st.bar_chart(df["class"].value_counts())

        st.subheader("🖼️ Sample Images")
        cols = st.columns(5)
        sample_imgs_per_class = {}

        for rec in records:
            the_class = rec["class"]
            if the_class not in sample_imgs_per_class and "error" not in rec:
                sample_imgs_per_class[the_class] = rec["file_name"]

        for idx, (cls, sample_img_name) in enumerate(sample_imgs_per_class.items()):
            found_img_path = None
            for dirpath, _, filenames in os.walk(dataset_root):
                if sample_img_name in filenames:
                    found_img_path = os.path.join(dirpath, sample_img_name)
                    break
            if found_img_path:
                try:
                    img = Image.open(found_img_path)
                    cols[idx % 5].image(img, caption=cls, use_container_width=True)
                except Exception as e:
                    cols[idx % 5].text(f"Error loading image: {e}")
    else:
        st.warning("⚠️ No supported image files found in dataset.")
elif dataset_root:
    st.error("❌ Invalid dataset path. Please enter a valid folder.")
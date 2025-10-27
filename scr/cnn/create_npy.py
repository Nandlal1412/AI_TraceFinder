import os
import numpy as np
from preprocess import load_and_preprocess
from feature_extraction import extract_noise

DATASET_DIR = "Datasets\Official"
FLATFIELD_DIR = "Datasets\Flatfield"
OUT_DIR = "processed_data/new_processed"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

def create_official_npy():
    X_img, y = [], []
    for label, device in enumerate(sorted(os.listdir(DATASET_DIR))):
        device_path = os.path.join(DATASET_DIR, device)
        if not os.path.isdir(device_path):
            continue  # skip non-folder files
        print(f"\nProcessing device folder: {device}")
        for res_folder in os.listdir(device_path):
            img_dir = os.path.join(device_path, res_folder)
            if not os.path.isdir(img_dir):
                continue
            print(f"  Processing resolution folder: {res_folder}")
            count = 0
            for img_file in os.listdir(img_dir):
                if not img_file.lower().endswith(".tif"):
                    continue
                img_path = os.path.join(img_dir, img_file)
                try:
                    img = load_and_preprocess(img_path)
                    X_img.append(img)
                    y.append(label)
                    count += 1
                    if count % 50 == 0:  # print every 50 images
                        print(f"    Processed {count} images...")
                except Exception as e:
                    print(f"    Skipping {img_path} due to error: {e}")
            print(f"  Completed {res_folder}: {count} images processed.")

    np.save(os.path.join(OUT_DIR, "official_images.npy"), np.array(X_img))
    np.save(os.path.join(OUT_DIR, "labels.npy"), np.array(y))
    print(f"Official images saved! Shape: {np.array(X_img).shape}, Labels: {len(y)}")


def create_flatfield_npy():
    X_noise, y = [], []
    for label, device in enumerate(sorted(os.listdir(FLATFIELD_DIR))):
        device_path = os.path.join(FLATFIELD_DIR, device)
        if not os.path.isdir(device_path):
            continue
        print(f"\nProcessing flatfield device folder: {device}")
        count = 0
        for img_file in os.listdir(device_path):
            if not img_file.lower().endswith(".tif"):
                continue
            img_path = os.path.join(device_path, img_file)
            try:
                noise = extract_noise(img_path)
                X_noise.append(noise)
                y.append(label)
                count += 1
                if count % 50 == 0:
                    print(f"    Processed {count} images...")
            except Exception as e:
                print(f"    Skipping {img_path} due to error: {e}")
        print(f"  Completed {device}: {count} images processed.")

    np.save(os.path.join(OUT_DIR, "flatfield_noise.npy"), np.array(X_noise))
    np.save(os.path.join(OUT_DIR, "flatfield_labels.npy"), np.array(y))
    print(f"Flatfield noise saved! Shape: {np.array(X_noise).shape}, Labels: {len(y)}")
if __name__ == "__main__":
    create_official_npy()
    create_flatfield_npy()

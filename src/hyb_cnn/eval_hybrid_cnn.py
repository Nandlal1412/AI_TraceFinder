import os, pickle, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Paths ----
ART_DIR   = "models"
MODEL_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")
RES_PATH = "Datasets/official_wiki_residuals.pkl"
FP_PATH  = "models/scanner_fingerprints.pkl"
ORDER_NPY = "models/fp_keys.npy"

# ---- Load label encoder + scaler ----
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Load residuals + fingerprints ----
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)
with open(FP_PATH, "rb") as f:
    scanner_fps = pickle.load(f)
fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()

# ---- Utilities ----
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]
    return feats

from skimage.feature import local_binary_pattern as sk_lbp
def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

# ---- Build test dataset ----
X_img_te, X_feat_te, y_te = [], [], []
for dataset_name in ["Official", "Wikipedia"]:
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                X_img_te.append(np.expand_dims(res,-1))
                v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                v_fft  = fft_radial_energy(res)
                v_lbp  = lbp_hist_safe(res)
                X_feat_te.append(v_corr + v_fft + v_lbp)
                y_te.append(scanner)

X_img_te = np.array(X_img_te, dtype=np.float32)
X_feat_te = np.array(X_feat_te, dtype=np.float32)
y_int_te = np.array([le.transform([c])[0] for c in y_te])

# Scale features
X_feat_te = scaler.transform(X_feat_te)

# ---- Evaluate ----
y_pred_prob = model.predict([X_img_te, X_feat_te])
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_int_te, y_pred)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print("\n✅ Classification Report:")
print(classification_report(y_int_te, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_int_te, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

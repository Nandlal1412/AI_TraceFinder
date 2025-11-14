# src/processing.py
import os, glob, csv
from pathlib import Path
from PIL import Image
import pymupdf as fitz
from tqdm import tqdm
from .utils import load_to_residual, extract_patches, make_feat_vector

def pdf_to_tiffs(pdf_path, out_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    stem = Path(pdf_path).stem
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        outp = os.path.join(out_dir, f"{stem}_p{i+1}.tif")
        img.save(outp, format="TIFF", dpi=(dpi, dpi))
    doc.close()

def build_tamper_manifest(root, out_csv):
    TAMP_ROOT = os.path.join(root, "TamperedImages")
    rows = [["path", "label", "domain", "tamper_type", "page_id"]]
    for dirpath, _, files in os.walk(TAMP_ROOT):
        for f in files:
            if not f.lower().endswith((".tif", ".tiff")): continue
            full = os.path.join(dirpath, f)
            label = 1 if "Tampered" in full else 0
            ttype = Path(dirpath).name
            rows.append([full, label, "tamper", ttype, Path(f).stem])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return out_csv

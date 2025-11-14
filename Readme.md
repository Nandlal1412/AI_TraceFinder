#  🕵️‍♂️ AI TraceFinder — Forensic Scanner Identification  

##  🧠 Project Overview  
Scanned documents—such as legal contracts, certificates, and financial records—are vulnerable to forgery. It’s often impossible to determine whether a scanned image is authentic or if it was created using an unauthorized device.

**AI TraceFinder** solves this challenge by identifying the **source scanner** used to digitize a document. Every scanner model introduces subtle, unique digital fingerprints—microscopic noise patterns, texture artifacts, and compression traces—based on its hardware and firmware. These invisible signatures can be used to:
- ✅ Attribute a scanned document to a specific scanner model
- 🚫 Detect forgeries created using unauthorized devices
- 🧾 Verify the authenticity of scanned evidence in forensic and legal contexts
  
Built as a forensic machine learning platform, AI TraceFinder extracts scanner-specific features and applies classification models to deliver **accurate, interpretable, and legally actionable results.** Whether for fraud detection, source attribution, or tamper validation, this tool empowers investigators with digital proof of origin.
  

---

##  🎯 Goals & Objectives  
- Collect and label scanned document datasets from multiple scanners  
- Robust preprocessing (resize, grayscale, normalize, denoise)  
- Extract scanner-specific features (noise, FFT, PRNU, texture descriptors)  
- Train classification models (ML + CNN)  
- Apply explainability tools (Grad-CAM, SHAP)  
- **Deploy an interactive app for scanner source identification**  
- Deliver **accurate, interpretable results** for forensic and legal use cases  

---

## 🛠 Tech Stack

This project leverages a modern stack for machine learning, image processing, and web application delivery.

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Backend & ML** | **Python** | Core programming language |
| | **Scikit-learn** | Random Forest & SVM (Baseline Models) |
| | **Pandas** | Data manipulation and CSV handling |
| | **OpenCV** | Image processing (loading, color conversion, etc.) |
| | **NumPy** | Numerical operations |
| | **TensorFlow / Keras** | For CNN Model |
| **Frontend & UI** | **Streamlit** | Creating the interactive web application |
| | **Matplotlib & Seaborn** | Data visualization (confusion matrix, plots) |
| | **Pillow (PIL)** | Displaying sample images in the UI |
| **Tooling** | **Git & GitHub** | Version control and source management |
| | **venv** | Python virtual environment management |


---

##  🧪 Methodology 
1. **Data Collection & Labeling**  
   - Gather scans from 3–5 scanner models/brands  
   - Create a structured, labeled dataset  

2. **Preprocessing**  
   - Resize, grayscale, normalize  
   - Optional: denoise to highlight artifacts  

3. **Feature Extraction**  
   - PRNU patterns, FFT, texture descriptors (LBP, edge features)  

4. **Model Training**  
   - Baseline ML: SVM, Random Forest, Logistic Regression  
   - Deep Learning: CNN with augmentation  

5. **Evaluation & Explainability**  
   - Metrics: Accuracy, F1-score, Confusion Matrix  
   - Interpretability: Grad-CAM, SHAP feature maps  

6. **Deployment**  
   - Streamlit app → upload scanned image → predict scanner model  
   - Display confidence score and key feature regions  

---

##  🕵️ Actionable Insights for Forensics  
- **Source Attribution:** Identify which scanner created a scanned copy of a document.  
- **Fraud Detection:** Detect forgeries where unauthorized scanners were used.  
- **Legal Verification:** Validate whether scanned evidence originated from approved devices.  
- **Tamper Resistance:** Differentiate between authentic vs. tampered scans.  
- **Explainability:** Provide visual evidence of how classification was made.  

---

##  🧱 Architecture (Conceptual)  
Input ➜ Preprocessing ➜ Feature Extraction + Modeling ➜ Evaluation & Explainability ➜ Prediction App  

---
## ✨ Key Capabilities
- **🔍 Flexible Feature Extraction:** Use the built-in Streamlit interface to process image folders, extract over 10 scanner-specific attributes, and automatically generate a structured CSV file.
- **📈 Interactive Data Insights:** Explore visual summaries including class balance charts, representative samples per scanner type, and a complete dataset preview.
- **📥 Exportable Analysis:** Instantly download the full feature dataset for offline modeling or archival purposes.
- **🧪 End-to-End ML Workflow:**
- **Model Training:** Train Random Forest and SVM classifiers directly from the extracted features.
- **Performance Review:** Access detailed metrics like precision, recall, and confusion matrices to evaluate model accuracy.
- **Real-Time Prediction:** Upload any scanned image to identify its originating scanner with confidence.
- **🔄 Model Selection Flexibility:** Toggle between SVM and Random Forest classifiers for prediction based on your use case.
- **🧠 Deep Learning Integration:** Leverage a convolutional neural network (CNN) for direct image-based classification, bypassing manual feature engineering.

---
## 🏠 Home Page :

![Home Page](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Home.png)


---
## 🗂️ Dataset Overview:

![Dataset Overview](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Dataset%20Overview.png)

---
## 🎨 Feature Visualization:

![Feature Visualization](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Feature%20Visualization.png)

---
## 📊 EDA

![EDA](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/EDA.png)

---

## 🧩 Feature Extraction:

![Feature Extraction](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Feature%20Extraction.png)

---
## 🧠 Model Training & Evaluation:

![Model Traning & Evaluation](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Model%20Training%20%26%20Evaluation.png)

---
## 🔍 Live Prediction:

![Live Prediction](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Live%20Prediction.png)

---
## 🚨 Forgery / Tampered Detection
![Forgery / Tampered Detection](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/Forgery_Tampered%20Detection.png)

---

## 🧑‍💻 About

![About](https://github.com/Nandlal1412/AI_TraceFinder/blob/main/interface_walkthrough/About.png)


---

## ⏳ 8-Week Roadmap (Milestones)  
- **W1:** Dataset collection (min. 3–5 scanners), labeling, metadata analysis  
- **W2:** Preprocessing pipeline (resize, grayscale, normalize, optional denoise)  
- **W3:** Feature extraction (noise maps, FFT, LBP, texture descriptors)  
- **W4:** Baseline ML models (SVM, RF, Logistic Regression) + evaluation  
- **W5:** CNN model training with augmentation, hyperparameter tuning  
- **W6:** Model evaluation (accuracy, F1, confusion matrix) + Grad-CAM/SHAP analysis  
- **W7:** Streamlit app development → image upload, prediction, confidence output  
- **W8:** Final documentation, results, presentation, and demo handover  

---



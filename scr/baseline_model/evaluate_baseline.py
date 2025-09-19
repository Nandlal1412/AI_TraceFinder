import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

CSV_PATH = "official.csv"

def evaluate_model(model_path, name, save_dir="results"):
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "class"])
    y = df["class"]

    # Load scaler + model
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(model_path)

    # Transform features
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Print report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Ensure results directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Confusion matrix saved to: {save_path}")

    # Show plot (optional, you can comment this out if not needed)
    plt.show()

if __name__ == "__main__":
    evaluate_model("models/random_forest.pkl", "Random Forest")
    evaluate_model("models/svm.pkl", "SVM")
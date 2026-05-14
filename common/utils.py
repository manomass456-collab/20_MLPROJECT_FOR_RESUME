from pathlib import Path
import json
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def project_paths(file_path):
    root = Path(file_path).resolve().parent
    model_dir = root / "models"
    report_dir = root / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    return root, model_dir, report_dir


def encode_categoricals(df):
    df = df.copy()
    encoders = {}
    for column in df.select_dtypes(include=["object", "category", "bool"]).columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder
    return df, encoders


def regression_report(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_model(model, path):
    joblib.dump(model, path)


def plot_feature_importance(model, columns, path, top_n=10):
    if not hasattr(model, "feature_importances_"):
        return
    pairs = sorted(zip(columns, model.feature_importances_), key=lambda item: item[1], reverse=True)[:top_n]
    names = [name for name, _ in pairs][::-1]
    scores = [score for _, score in pairs][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(names, scores)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

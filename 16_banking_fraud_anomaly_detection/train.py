from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import encode_categoricals, project_paths, save_json, save_model

TARGET = "anomaly"


def main():
    root, model_dir, report_dir = project_paths(__file__)
    df = pd.read_csv(root / "dataset.csv")
    encoded, encoders = encode_categoricals(df)
    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(n_estimators=180, contamination=0.08, random_state=42))])
    raw_predictions = pipeline.fit_predict(X)
    predictions = [1 if value == -1 else 0 for value in raw_predictions]
    metrics = {"rows": int(len(df)), "features": list(X.columns), "classification_report_against_synthetic_labels": classification_report(y, predictions, output_dict=True, zero_division=0)}
    save_json(metrics, report_dir / "metrics.json")
    save_model({"pipeline": pipeline, "encoders": encoders, "features": list(X.columns)}, model_dir / "model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import classification_report_dict, encode_categoricals, plot_feature_importance, project_paths, save_json, save_model

TARGET = "intrusion"


def main():
    root, model_dir, report_dir = project_paths(__file__)
    df = pd.read_csv(root / "dataset.csv")
    encoded, encoders = encode_categoricals(df)
    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = classification_report_dict(y_test, predictions)
    metrics["rows"] = int(len(df))
    metrics["features"] = list(X.columns)
    save_json(metrics, report_dir / "metrics.json")
    save_model({"model": model, "encoders": encoders, "features": list(X.columns)}, model_dir / "model.joblib")
    plot_feature_importance(model, X.columns, report_dir / "feature_importance.png")
    print(metrics)


if __name__ == "__main__":
    main()

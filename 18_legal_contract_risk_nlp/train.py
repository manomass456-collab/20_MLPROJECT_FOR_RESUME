from pathlib import Path
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import classification_report_dict, encode_categoricals, project_paths, save_json, save_model

TARGET = "risk_label"
TEXT_COLUMN = "clause_text"


def main():
    root, model_dir, report_dir = project_paths(__file__)
    df = pd.read_csv(root / "dataset.csv")
    labels, encoders = encode_categoricals(df[[TARGET]])
    X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COLUMN], labels[TARGET], test_size=0.25, random_state=42, stratify=labels[TARGET])
    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = classification_report_dict(y_test, predictions)
    metrics["rows"] = int(len(df))
    metrics["text_column"] = TEXT_COLUMN
    save_json(metrics, report_dir / "metrics.json")
    save_model({"pipeline": pipeline, "encoders": encoders}, model_dir / "model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()

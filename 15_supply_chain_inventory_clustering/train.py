from pathlib import Path
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import encode_categoricals, project_paths, save_json, save_model


def main():
    root, model_dir, report_dir = project_paths(__file__)
    df = pd.read_csv(root / "dataset.csv")
    encoded, encoders = encode_categoricals(df.drop(columns=["known_segment"], errors="ignore"))
    pipeline = Pipeline([("scaler", StandardScaler()), ("cluster", KMeans(n_clusters=3, random_state=42, n_init=20))])
    labels = pipeline.fit_predict(encoded)
    df["predicted_cluster"] = labels
    df.to_csv(report_dir / "clustered_inventory.csv", index=False)
    metrics = {"silhouette_score": float(silhouette_score(encoded, labels)), "rows": int(len(df)), "features": list(encoded.columns), "cluster_counts": df["predicted_cluster"].value_counts().sort_index().to_dict()}
    save_json(metrics, report_dir / "metrics.json")
    save_model({"pipeline": pipeline, "encoders": encoders, "features": list(encoded.columns)}, model_dir / "model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()

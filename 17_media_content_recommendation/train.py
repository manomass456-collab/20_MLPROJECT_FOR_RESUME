from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.utils import encode_categoricals, project_paths, save_json, save_model

TARGET = "engagement_score"


def recommend_similar_items(item_index, feature_matrix, n=5):
    similarities = cosine_similarity(feature_matrix[item_index:item_index + 1], feature_matrix).ravel()
    ranked = similarities.argsort()[::-1]
    return [int(idx) for idx in ranked if idx != item_index][:n]


def main():
    root, model_dir, report_dir = project_paths(__file__)
    df = pd.read_csv(root / "dataset.csv")
    encoded, encoders = encode_categoricals(df)
    X = encoded.drop(columns=[TARGET])
    scaler = StandardScaler()
    matrix = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn.fit(matrix)
    sample_recommendations = {str(i): recommend_similar_items(i, matrix, n=5) for i in range(5)}
    metrics = {"rows": int(len(df)), "features": list(X.columns), "sample_recommendations": sample_recommendations}
    save_json(metrics, report_dir / "metrics.json")
    save_model({"scaler": scaler, "nearest_neighbors": nn, "encoders": encoders, "features": list(X.columns)}, model_dir / "model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()

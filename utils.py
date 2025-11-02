from sentence_transformers import SentenceTransformer  
import numpy as np
import json
from pathlib import Path

def load_scenario(path: str) -> dict:
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
    
text_embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def extract_features_from_query(X: list, y: str, dataset_columns: list, threshold: float = 0.55):
    X_lower = [x.strip().lower() for x in X]
    y_lower = y.strip().lower()
    cols_lower = [c.strip().lower() for c in dataset_columns]

    emb_features = text_embedding_model.encode(X_lower + [y_lower])
    emb_columns  = text_embedding_model.encode(cols_lower)

    sim_matrix = cosine_sim(emb_features, emb_columns)

    mapping_results = {}
    for i, name in enumerate(X_lower + [y_lower]):
        sims = sim_matrix[i]
        best_idx = np.argmax(sims)
        best_col = dataset_columns[best_idx]
        best_score = sims[best_idx]
        mapping_results[name] = {
            "matched_column": best_col if best_score >= threshold else None,
            "score": float(best_score)
        }

    mapped_X = [mapping_results[x]["matched_column"] for x in X_lower]
    mapped_y = mapping_results[y_lower]["matched_column"]

    for k, v in mapping_results.items():
        print(f"{k:>20s} -> {v['matched_column']}  (score={v['score']:.3f})")

    return mapped_X, mapped_y, mapping_results
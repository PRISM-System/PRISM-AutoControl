from __future__ import annotations
from typing import Dict, Any, List, Optional
import pathlib
import numpy as np
import pandas as pd

from utils import predict_expected_y as fallback_predict_expected_y

LINEAR_KIND = "linear_simple_v1"

# -------------------------------
# 학습(OLS, 전행 사용, 전처리 최소)
# -------------------------------
def train_simple_linear(
    feature_df: pd.DataFrame,
    target_col: str,
    x_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if target_col not in feature_df.columns:
        raise ValueError(f"target_col '{target_col}' not in feature_df")

    if x_cols is None:
        x_cols = [c for c in feature_df.columns if c != target_col]

    X = feature_df[x_cols].to_numpy(dtype=float, copy=True)
    y = feature_df[target_col].to_numpy(dtype=float, copy=True)

    # 간단 방어: NaN/Inf → 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # 편향항 추가: [X | 1]
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([X, ones])

    # OLS: w_full = (Xb^T Xb)^(-1) Xb^T y  (pinv로 안정화)
    w_full = np.linalg.pinv(Xb) @ y
    w = w_full[:-1]           # 가중치
    b = float(w_full[-1])     # bias

    model = {
        "kind": LINEAR_KIND,
        "target_col": target_col,
        "feature_names": x_cols,
        "weights": w.astype(float),    # shape (d,)
        "bias": b                      # scalar
    }
    return model

def save_linear_model(model: Dict[str, Any], save_path: pathlib.Path) -> pathlib.Path:
    save_path = pathlib.Path(save_path).with_suffix(".npz")
    np.savez(
        save_path,
        kind=np.array(model["kind"]),
        target_col=np.array(model["target_col"]),
        feature_names=np.array(model["feature_names"], dtype=object),
        weights=model["weights"],
        bias=np.array(model["bias"]),
    )
    return save_path.resolve()

def load_linear_model(path: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=True) as npz:
        kind = str(npz["kind"])
        if kind != LINEAR_KIND:
            raise ValueError(f"unsupported model kind: {kind}")
        model = {
            "kind": kind,
            "target_col": str(npz["target_col"]),
            "feature_names": list(npz["feature_names"].tolist()),
            "weights": npz["weights"].astype(float),
            "bias": float(npz["bias"]),
        }
    return model

# -------------------------------
# 예측 함수: expected_y_fn(last_y, delta_y, adjustments, horizon, feature_df=..., target_col=...)
# -------------------------------
def make_expected_y_fn_from_linear(model: Dict[str, Any]):
    x_cols: List[str] = model["feature_names"]
    w: np.ndarray = model["weights"]
    b: float = model["bias"]

    def expected_y_fn(
        last_y: float,
        delta_y: float,
        adjustments: Dict[str, float],
        horizon: int,
        *,
        feature_df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None
    ) -> float:
        # feature_df 없으면 폴백
        if feature_df is None:
            return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))

        # 마지막 행을 baseline으로, adjustments 적용
        try:
            base_row = feature_df.tail(1)[x_cols].iloc[0].to_dict()
        except Exception:
            return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))

        x_vec = np.array(
            [float(base_row.get(c, 0.0) + adjustments.get(c, 0.0)) for c in x_cols],
            dtype=float
        )
        y_hat = float(x_vec @ w + b)
        return y_hat

    return expected_y_fn

# -------------------------------
# 고수준 API (학습 후 저장)
# -------------------------------
def train_and_save_linear_simple(
    feature_df: pd.DataFrame,
    target_col: str,
    save_dir: pathlib.Path,
    *,
    x_cols: Optional[List[str]] = None,
    filename_stub: str = "linear_surrogate"
) -> pathlib.Path:
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = train_simple_linear(feature_df, target_col, x_cols=x_cols)
    path = save_dir / f"{filename_stub}.npz"
    return save_linear_model(model, path)

# -------------------------------
# 로더(가중치 경로가 선형모델이면 → 선형, 아니면 폴백)
# -------------------------------
def load_surrogate(weight_local_path: Optional[str]):
    def fallback_fn(last_y, delta_y, adjustments, horizon, **kwargs) -> float:
        return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))

    if not weight_local_path:
        return fallback_fn

    try:
        model = load_linear_model(pathlib.Path(weight_local_path))
        return make_expected_y_fn_from_linear(model)
    except Exception:
        return fallback_fn
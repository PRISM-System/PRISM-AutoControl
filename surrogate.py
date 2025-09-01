from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import pathlib
import numpy as np
import pandas as pd

from utils import predict_expected_y as fallback_predict_expected_y

try:
    from joblib import dump as _joblib_dump, load as _joblib_load
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    import sklearn  
    from sklearn.base import ClassifierMixin, RegressorMixin 
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

LINEAR_KIND = "linear_simple_v1"

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

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([X, ones])
    w_full = np.linalg.pinv(Xb) @ y
    w = w_full[:-1]
    b = float(w_full[-1])

    model = {
        "kind": LINEAR_KIND,
        "target_col": target_col,
        "feature_names": x_cols,
        "weights": w.astype(float),
        "bias": b
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
        if feature_df is None:
            return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))
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

_SKLEARN_REGISTRY = {
    # 회귀
    "linear_regression": ("sklearn.linear_model", "LinearRegression", "regression"),
    "ridge": ("sklearn.linear_model", "Ridge", "regression"),
    "lasso": ("sklearn.linear_model", "Lasso", "regression"),
    "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor", "regression"),
    "gradient_boosting_regressor": ("sklearn.ensemble", "GradientBoostingRegressor", "regression"),
    "svr": ("sklearn.svm", "SVR", "regression"),
    # 분류
    "logistic_regression": ("sklearn.linear_model", "LogisticRegression", "classification"),
    "random_forest_classifier": ("sklearn.ensemble", "RandomForestClassifier", "classification"),
    "gradient_boosting_classifier": ("sklearn.ensemble", "GradientBoostingClassifier", "classification"),
    "svc": ("sklearn.svm", "SVC", "classification"),
}

def _import_sklearn_estimator(module_name: str, cls_name: str):
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, cls_name)

def train_and_save_sklearn(
    feature_df: pd.DataFrame,
    target_col: str,
    save_dir: pathlib.Path,
    *,
    x_cols: Optional[List[str]] = None,
    model: Union[str, Dict[str, Any]] = "random_forest_regressor",
    filename_stub: str = "sk_surr",
) -> pathlib.Path:
    if not (_HAS_SKLEARN and _HAS_JOBLIB):
        raise RuntimeError("scikit-learn/joblib이 설치되어 있지 않습니다. (pip install scikit-learn joblib)")

    if x_cols is None:
        x_cols = [c for c in feature_df.columns if c != target_col]

    X = feature_df[x_cols].to_numpy(dtype=float, copy=True)
    y = feature_df[target_col].copy()

    if isinstance(model, str):
        if model not in _SKLEARN_REGISTRY:
            raise ValueError(f"unknown sklearn model key: {model}")
        mod_name, cls_name, problem_type = _SKLEARN_REGISTRY[model]
        Est = _import_sklearn_estimator(mod_name, cls_name)
        est = Est()
        class_opt = None
        model_name = model
        params = {}
    else:
        name = model.get("name")
        if name not in _SKLEARN_REGISTRY:
            raise ValueError(f"unknown sklearn model key: {name}")
        mod_name, cls_name, problem_type = _SKLEARN_REGISTRY[name]
        Est = _import_sklearn_estimator(mod_name, cls_name)
        params = model.get("params") or {}
        est = Est(**params)
        class_opt = model.get("class_opt")
        model_name = name

    est.fit(X, y)

    meta = {
        "kind": "sklearn",
        "problem_type": problem_type,  # "regression" | "classification"
        "target_col": target_col,
        "feature_names": x_cols,
        "model_name": model_name,
        "params": params,
    }
    # 분류의 경우 클래스 정보/양성 클래스 결정
    if hasattr(est, "classes_"):
        classes = list(getattr(est, "classes_"))
        meta["classes"] = classes
        if class_opt is None:
            # 이진 분류: classes_[1]을 양성으로 간주. 멀티는 최댓값 확률을 반환(=class_opt 미사용)
            if len(classes) == 2:
                class_opt = classes[1]
        meta["class_opt"] = class_opt

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{filename_stub}.joblib"

    payload = {
        "meta": meta,
        "estimator": est,
    }
    _joblib_dump(payload, path)
    return path.resolve()

def _make_expected_y_fn_from_sklearn(payload: Dict[str, Any]):
    est = payload["estimator"]
    meta = payload["meta"]
    x_cols: List[str] = meta["feature_names"]
    problem_type: str = meta["problem_type"]
    class_opt = meta.get("class_opt")

    def expected_y_fn(
        last_y: float,
        delta_y: float,
        adjustments: Dict[str, float],
        horizon: int,
        *,
        feature_df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None
    ) -> float:
        if feature_df is None:
            return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))
        try:
            base_row = feature_df.tail(1)[x_cols].iloc[0].to_dict()
        except Exception:
            return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))

        x_vec = np.array(
            [[float(base_row.get(c, 0.0) + adjustments.get(c, 0.0)) for c in x_cols]],
            dtype=float
        )

        if problem_type == "regression":
            y_hat = float(est.predict(x_vec).reshape(-1)[0])
            return y_hat

        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(x_vec)[0]
            if class_opt is not None:
                try:
                    idx = list(est.classes_).index(class_opt)
                    return float(proba[idx])
                except Exception:
                    pass
            return float(np.max(proba))
        # prob이 없을 때: decision_function -> 시그모이드 근사
        if hasattr(est, "decision_function"):
            score = float(est.decision_function(x_vec).reshape(-1)[0])
            return float(1.0 / (1.0 + np.exp(-score)))
        y_pred = est.predict(x_vec).reshape(-1)[0]
        try:
            return float(y_pred)
        except Exception:
            return 1.0 if str(y_pred) == str(class_opt) else 0.0

    return expected_y_fn

def load_surrogate(weight_local_path: Optional[str]):
    def fallback_fn(last_y, delta_y, adjustments, horizon, **kwargs) -> float:
        return float(fallback_predict_expected_y(last_y, delta_y, adjustments, horizon))

    if not weight_local_path:
        return fallback_fn

    p = pathlib.Path(weight_local_path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        try:
            model = load_linear_model(p)
            return make_expected_y_fn_from_linear(model)
        except Exception:
            return fallback_fn

    if suffix in {".joblib", ".pkl"} and _HAS_JOBLIB:
        try:
            payload = _joblib_load(p)
            if isinstance(payload, dict) and "estimator" in payload and "meta" in payload:
                return _make_expected_y_fn_from_sklearn(payload)
        except Exception:
            pass

    return fallback_fn

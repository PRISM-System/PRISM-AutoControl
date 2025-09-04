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
    y_hat_train = (X @ w) + b
    mse = float(((y - y_hat_train) ** 2).mean())
    # R^2 수식: 1 - SSE/SST (SST가 0이면 0으로)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - float(((y - y_hat_train) ** 2).sum()) / sst) if sst > 0 else 0.0

    model = {
        "kind": LINEAR_KIND,
        "target_col": target_col,
        "feature_names": x_cols,
        "weights": w.astype(float),
        "bias": b,
        "train_metrics": {"mse": mse, "r2": r2}  # <-- 여기
    }
    return model

def save_linear_model(model: Dict[str, Any], save_path: pathlib.Path) -> pathlib.Path:
    save_path = pathlib.Path(save_path).with_suffix(".npz")
    # dict는 JSON으로 저장
    import json
    np.savez(
        save_path,
        kind=np.array(model["kind"]),
        target_col=np.array(model["target_col"]),
        feature_names=np.array(model["feature_names"], dtype=object),
        weights=model["weights"],
        bias=np.array(model["bias"]),
        train_metrics=np.array(json.dumps(model.get("train_metrics", {})))  # <-- ADD
    )
    return save_path.resolve()

def load_linear_model(path: pathlib.Path) -> Dict[str, Any]:
    import json
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
        # === ADD: restore metrics if present ===
        if "train_metrics" in npz:
            try:
                model["train_metrics"] = json.loads(str(npz["train_metrics"]))
            except Exception:
                model["train_metrics"] = {}
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

    # === ADD: train metrics on training set ===
    try:
        y_hat = est.predict(X)
        # 분류면 확률/결정값이 아닌 label-pred로 대체
        if problem_type == "classification" and hasattr(est, "predict"):
            # 분류 정확도도 함께 계산(선택)
            from sklearn.metrics import accuracy_score
            acc = float(accuracy_score(y, y_hat))
            train_metrics = {"acc": acc}
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            train_metrics = {
                "mse": float(mean_squared_error(y, y_hat)),
                "r2":  float(r2_score(y, y_hat))
            }
    except Exception:
        train_metrics = {}

    meta = {
        "kind": "sklearn",
        "problem_type": problem_type,  # "regression" | "classification"
        "target_col": target_col,
        "feature_names": x_cols,
        "model_name": model_name,
        "params": params,
        "train_metrics": train_metrics,  # <-- ADD
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

def _make_expected_y_fn_from_sklearn(payload):
    est = payload["estimator"]
    meta = payload.get("meta", {})
    feat_names = meta.get("feature_names")  # 학습시 사용한 피처 순서 그대로

    def fn(last_y, delta_y, adjs, horizon, *, feature_df=None, target_col=None):
        # 1) 베이스 벡터: feature_df 최신행에서 feat_names 순서대로 추출
        if feature_df is None or feat_names is None:
            # 안전장치: 조정값 합만 쓰는 휴리스틱으로 폴백(임시)
            return float(last_y + sum(adjs.values()) * 0.6 * max(min(horizon/20.0,1.2),0.3) - 0.1*delta_y)

        base_row = feature_df.iloc[-1]  # 최신
        x = []
        for f in feat_names:
            val = float(base_row.get(f, 0.0))
            if f in adjs:
                val += float(adjs[f])  # 조정 적용(절대값 덧셈)
            x.append(val)

        # 2) (선택) 스케일러/전처리 있으면 여기서 적용
        # ex) if "scaler" in meta: x = meta["scaler"].transform([x])[0]

        # 3) 예측
        yhat = est.predict([x])[0]
        # (선택) 역변환/meta 후처리 있으면 여기서 적용

        return float(yhat)

    return fn

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
def debug_check_roundtrip(weight_local_path: str,
                          df: pd.DataFrame,
                          feature_names: List[str],
                          target_col: str,
                          horizon: int = 1) -> Dict[str, Any]:
    """
    저장된 모델을 load_surrogate로 불러 expected_y_fn을 만들고,
    몇 가지 adjustments에 대해 예측이 잘 나오는지 점검한다.
    또한 저장물에 내장된 train_metrics(있다면)도 리턴한다.
    """
    exp_fn = load_surrogate(weight_local_path)

    # baseline & 간단한 변화 테스트
    last_y = float(df[target_col].iloc[-1])
    delta_y = float(df[target_col].iloc[-1] - df[target_col].iloc[-2]) if len(df) >= 2 else 0.0

    def _pred(adj):
        return float(exp_fn(last_y, delta_y, adj, horizon, feature_df=df, target_col=target_col))

    zero = {f: 0.0 for f in feature_names}
    one_up = {feature_names[0]: 1.0, **{f: 0.0 for f in feature_names[1:]}}
    res = {
        "predict_zero": _pred(zero),
        "predict_first_feature_plus1": _pred(one_up),
    }

    # 저장 파일 내부 metrics(가능한 경우)도 같이 추출
    try:
        p = pathlib.Path(weight_local_path)
        if p.suffix.lower() == ".npz":
            m = load_linear_model(p)
            res["train_metrics"] = m.get("train_metrics")
        elif p.suffix.lower() in {".joblib", ".pkl"} and _HAS_JOBLIB:
            payload = _joblib_load(p)
            tm = (payload.get("meta") or {}).get("train_metrics")
            res["train_metrics"] = tm
    except Exception:
        pass

    return res
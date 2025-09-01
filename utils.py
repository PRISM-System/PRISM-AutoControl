import os, pathlib, hashlib, base64, requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def rid() -> str:
    return f"req_{os.urandom(4).hex()}"

def ensure_dirs(paths: List[pathlib.Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def save_result_csv(rows: List[Dict[str, Any]], out_dir: pathlib.Path, task_id: str, ac_id: str) -> pathlib.Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{task_id}_{ac_id}_{ts}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path.resolve()

def artifact_download_url(filename: str) -> str:
    return f"/api/v1/autocontrol/artifacts/result?file={filename}"

def load_csv_and_features(
    csv_path: Optional[str],
    default_csv_path: pathlib.Path,
    *,
    numeric_only: bool = True,
) -> Tuple[pathlib.Path, pd.DataFrame, pd.DataFrame, List[str]]:
    csv_file = pathlib.Path(csv_path).expanduser().resolve() if csv_path else pathlib.Path(default_csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    if numeric_only:
        feature_df = df.select_dtypes(include=["number"]).copy()
    else:
        feature_df = df.copy()

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No numeric columns detected in CSV. "
            "Ensure your CSV has numeric features or disable numeric_only."
        )

    csv_features = list(feature_df.columns)
    return csv_file, df, feature_df, csv_features

def ensure_target_in_features(feature_df: pd.DataFrame, target_col: str):
    if target_col not in feature_df.columns:
        preview = list(feature_df.columns)[:10]
        raise ValueError(
            f"target_col '{target_col}' not found among numeric columns. "
            f"Numeric columns preview: {preview}{' ...' if feature_df.shape[1] > 10 else ''}"
        )

def weights_from_url(url: str, weights_cache_dir: pathlib.Path, timeout: int = 20) -> pathlib.Path:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.content
    h = hashlib.sha256(data).hexdigest()[:16]
    path = weights_cache_dir / f"ckpt_{h}.bin"
    with open(path, "wb") as f:
        f.write(data)
    return path.resolve()

def weights_from_b64(b64: str, weights_cache_dir: pathlib.Path) -> pathlib.Path:
    blob = base64.b64decode(b64)
    h = hashlib.sha256(blob).hexdigest()[:16]
    path = weights_cache_dir / f"ckpt_{h}.bin"
    with open(path, "wb") as f:
        f.write(blob)
    return path.resolve()

def fetch_pred_activation(base_url: str, task_id: str, inline_base64: bool = True) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/prediction/activate_autocontrol/{task_id}?inline_base64={'true' if inline_base64 else 'false'}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "SUCCESS":
        raise RuntimeError(f"prediction activation failed: {j}")
    return j.get("data") or {}

def ingest_weights(
    *,
    weight_path: Optional[str],
    weight_url: Optional[str],
    weights_b64: Optional[str],
    pred_activation_base_url: Optional[str],
    pred_task_id: Optional[str],
    weights_cache_dir: pathlib.Path,
    events: List[str],
) -> Dict[str, Any]:
    def _note(msg: str): events.append(f"[{now_iso()}] {msg}")

    # 1) base64
    if weights_b64:
        try:
            p = weights_from_b64(weights_b64, weights_cache_dir)
            _note(f"[AC] weights loaded from base64: {p.name}")
            return {"weight_local_path": str(p)}
        except Exception as e:
            _note(f"[AC] base64 weight failed: {e}")
    # 2) URL
    if weight_url:
        try:
            p = weights_from_url(weight_url, weights_cache_dir)
            _note(f"[AC] weights downloaded from url: {p.name}")
            return {"weight_local_path": str(p)}
        except Exception as e:
            _note(f"[AC] weight_url failed: {e}")
    # 3) 로컬 경로
    if weight_path:
        p = pathlib.Path(weight_path).expanduser().resolve()
        if p.exists():
            _note(f"[AC] weights from path: {p.name}")
            return {"weight_local_path": str(p)}
        else:
            _note(f"[AC] weight_path not found: {p}")
    # 4) prediction activation
    if pred_activation_base_url and pred_task_id:
        try:
            data = fetch_pred_activation(pred_activation_base_url, pred_task_id, inline_base64=True)
            if data.get("weight_inlined") and data.get("weight_base64"):
                p = weights_from_b64(data["weight_base64"], weights_cache_dir)
                _note(f"[AC] weights fetched via prediction activation (base64): {p.name}")
                return {"weight_local_path": str(p), "best_model": data.get("best_model")}
            if data.get("weight_path"):
                _note(f"[AC] prediction activation provided weight_path (remote): {data['weight_path']}")
                return {"weight_remote_hint": data.get("weight_path"), "best_model": data.get("best_model")}
        except Exception as e:
            _note(f"[AC] prediction activation fetch failed: {e}")
    return {}

def apply_constraints(adjustments: Dict[str, float], constraints: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not constraints or not isinstance(constraints, dict):
        return adjustments
    out = dict(adjustments)
    bounds = constraints.get("bounds") or {}
    for k, b in bounds.items():
        if k in out:
            lo = float(b.get("min", out[k]))
            hi = float(b.get("max", out[k]))
            out[k] = max(min(out[k], hi), lo)
            step = b.get("step")
            if step:
                st = float(step)
                out[k] = round(round(out[k] / st) * st, 10)
    return out
def predict_expected_y(last_y: float, delta_y: float, adjustments: Dict[str, float], horizon: int) -> float:
    gain = 0.6
    horizon_factor = min(max(horizon / 20.0, 0.3), 1.2)
    return last_y + sum(adjustments.values()) * gain * horizon_factor + (-0.1 * delta_y)

def score_candidate(expected_y: float, setpoint: float, adjustments: Dict[str, float]) -> float:
    gap = abs(expected_y - setpoint)
    l1  = sum(abs(v) for v in adjustments.values())
    return float(gap + 0.3 * l1)

import os, pathlib, hashlib, base64, requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dirs(paths: List[pathlib.Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

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
    return float(gap + 0 * l1)
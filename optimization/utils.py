import pandas as pd
import numpy as np

# 공정 데이터의 샘플링 step 추출 (최근 10샘플의 샘플링 주기 평균)
def infer_dt_seconds(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df['TIMESTAMP'])
    dt = ts.diff().dt.total_seconds().dropna().tail(10).mean()
    return float(dt if np.isfinite(dt) and dt > 0 else 10.0)
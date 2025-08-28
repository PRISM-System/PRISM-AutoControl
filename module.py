from typing import Any, Dict, List
import numpy as np

def risk_module(feature_df, target_col: str, cand_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    후보마다 expected_y와 setpoint_hint 차이를 보고 간단 위험레벨 산출.
    summary.volatility는 expected_y의 표준편차.
    """
    by = []
    if not cand_dicts:
        return {"byCandidate": by, "summary": {"volatility": 0.0}}
    ys = [float(c.get("expected_y", 0.0)) for c in cand_dicts]
    vol = float(np.std(ys)) if len(ys) > 1 else 0.0
    for c in cand_dicts:
        ey = float(c.get("expected_y", 0.0))
        sp = float(c.get("setpoint_hint", ey))
        gap = abs(ey - sp)
        lvl = "high" if gap > 5 else ("medium" if gap > 2 else "low")
        by.append({"id": c.get("id"), "expected_y": ey, "level": lvl, "gap": round(gap, 6)})
    return {"byCandidate": by, "summary": {"volatility": round(vol, 6)}}

def explain_module(feature_df, target_col: str, cand_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    후보들의 adjustments 절댓값 합을 변수 중요도로 정규화.
    """
    acc: Dict[str, float] = {}
    for c in cand_dicts:
        for k, v in (c.get("adjustments") or {}).items():
            acc[k] = acc.get(k, 0.0) + abs(float(v))
    tot = sum(acc.values()) or 1.0
    imps = [{"name": k, "importance": round(v / tot, 4)}
            for k, v in sorted(acc.items(), key=lambda x: -x[1])]
    return {"importantFeatures": imps, "method": "Σ|Δx| normalized across candidates"}

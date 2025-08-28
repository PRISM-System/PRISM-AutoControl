# optimizer.py — 제어 입력 x 최적화 (좌표탐색 + 랜덤 리스타트)
from typing import Dict, Any, List, Optional, Tuple
import math, random

from utils import apply_constraints, predict_expected_y, score_candidate

# 기본 바운드/스텝 (제약 미지정 시)
_DEFAULT_BOUNDS = (-0.5, 0.5)
_DEFAULT_STEPS  = 0.05

def _norm_step(lo: float, hi: float, step: Optional[float]) -> float:
    if step and step > 0:
        return float(step)
    span = max(hi - lo, 1e-6)
    return round(span / 20.0, 10)

def _round_to_step(x: float, step: float) -> float:
    return round(round(x / step) * step, 10)

def _make_bounds(variables: List[str], constraints: Optional[Dict[str, Any]]) -> Dict[str, Tuple[float,float,float]]:
    bounds_cfg = (constraints or {}).get("bounds", {}) if isinstance(constraints, dict) else {}
    out: Dict[str, Tuple[float,float,float]] = {}
    for v in variables:
        b = bounds_cfg.get(v, {})
        lo = float(b.get("min", _DEFAULT_BOUNDS[0]))
        hi = float(b.get("max", _DEFAULT_BOUNDS[1]))
        st = _norm_step(lo, hi, b.get("step"))
        out[v] = (lo, hi, st)
    return out

def _objective(expected_y: float, setpoint: float, adjustments: Dict[str, float]) -> float:
    # score와 동일한 형태로 목표화
    return score_candidate(expected_y, setpoint, adjustments)

def optimize_control(
    *,
    variables: List[str],                 # 조작변수 이름들 (MV)
    last_y: float, delta_y: float,        # 최근 타깃 y 상태
    setpoint: float, horizon: int,        # 목표/구간
    constraints: Optional[Dict[str, Any]],
    top_k: int = 1,
    # 서러게이트 예측 함수 (옵션): 시그니처는 아래와 동일
    expected_y_fn = None,                 # fn(last_y, delta_y, adjustments, horizon, feature_df=None, target_col=None) -> float
    feature_df=None,
    target_col: Optional[str] = None,
    # 탐색 하이퍼파라미터
    restarts: int = 12,
    iters_per_restart: int = 80,
) -> List[Dict[str, Any]]:

    bounds = _make_bounds(variables, constraints)

    def clamp_round_vec(adjs: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x = max(min(adjs.get(v, 0.0), hi), lo)
            out[v] = _round_to_step(x, st)
        return out

    def predict(adjs: Dict[str, float]) -> float:
        if expected_y_fn:
            return float(expected_y_fn(last_y, delta_y, adjs, horizon, feature_df=feature_df, target_col=target_col))
        # 폴백: 간이 동역학
        return float(predict_expected_y(last_y, delta_y, adjs, horizon))

    # 초기화 후보 생성기
    def random_init() -> Dict[str, float]:
        adjs = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x = random.uniform(lo, hi)
            adjs[v] = _round_to_step(x, st)
        return adjs

    results: List[Dict[str, Any]] = []

    for _ in range(restarts):
        # 0) 시작점: 0 또는 랜덤
        cur = {v: 0.0 for v in variables}
        # gap이 크면 랜덤도 한 번 써봄
        if abs(setpoint - last_y) > 1.0:
            cur = random_init()
        cur = clamp_round_vec(cur)

        ey = predict(cur)
        cur_score = _objective(ey, setpoint, cur)

        for _iter in range(iters_per_restart):
            improved = False
            # 좌표 탐색
            for v in variables:
                lo, hi, st = bounds[v]
                base = cur[v]
                # 세 방향 평가: 그대로 / +step / -step
                candidates = [base, base + st, base - st]
                best_v = base
                best_score = cur_score
                best_ey = ey
                for cand in candidates:
                    cand = max(min(cand, hi), lo)
                    cand = _round_to_step(cand, st)
                    if cand == base:
                        eyy = ey
                        sc = cur_score
                    else:
                        trial = dict(cur)
                        trial[v] = cand
                        eyy = predict(trial)
                        sc = _objective(eyy, setpoint, trial)
                    if sc < best_score - 1e-9:
                        best_score, best_v, best_ey = sc, cand, eyy

                if best_v != base:
                    cur[v] = best_v
                    cur_score = best_score
                    ey = best_ey
                    improved = True

            if not improved:
                break  # 정체 시 종료

        results.append({"adjustments": dict(cur), "expected_y": round(float(ey), 6), "score": round(float(cur_score), 6)})

    # 중복 제거(조정벡터가 같은 것 합치기) + 정렬
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in results:
        key = "|".join(f"{k}:{r['adjustments'][k]:.6f}" for k in variables)
        if key not in uniq or r["score"] < uniq[key]["score"]:
            uniq[key] = r
    ranked = sorted(uniq.values(), key=lambda x: x["score"])

    # top-k로 후보아이디 부여
    out = []
    for i, r in enumerate(ranked[:top_k], start=1):
        out.append({
            "id": f"cand_{i}",
            "adjustments": r["adjustments"],
            "expected_y": r["expected_y"],
            "score": r["score"]
        })
    return out

from typing import Dict, Any, List, Optional, Tuple
import math, random

from utils import apply_constraints, predict_expected_y, score_candidate

try:
    import numpy as _np
    from scipy import optimize as _spopt 
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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
    return score_candidate(expected_y, setpoint, adjustments)

def _predict_wrapper(expected_y_fn, last_y, delta_y, horizon, feature_df, target_col):
    def _predict(adjs: Dict[str, float]) -> float:
        if expected_y_fn:
            return float(expected_y_fn(last_y, delta_y, adjs, horizon, feature_df=feature_df, target_col=target_col))
        return float(predict_expected_y(last_y, delta_y, adjs, horizon))
    return _predict

def _coordinate_search(
    variables, bounds, predict, setpoint,
    restarts=12, iters_per_restart=80
):
    def clamp_round_vec(adjs: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x = max(min(adjs.get(v, 0.0), hi), lo)
            out[v] = _round_to_step(x, st)
        return out

    def random_init() -> Dict[str, float]:
        adjs = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x = random.uniform(lo, hi)
            adjs[v] = _round_to_step(x, st)
        return adjs

    results: List[Dict[str, Any]] = []
    for _ in range(restarts):
        cur = {v: 0.0 for v in variables}
        cur = clamp_round_vec(cur)
        ey = predict(cur)
        cur_score = _objective(ey, setpoint, cur)

        for _iter in range(iters_per_restart):
            improved = False
            for v in variables:
                lo, hi, st = bounds[v]
                base = cur[v]
                candidates = [base, base + st, base - st]
                best_v, best_score, best_ey = base, cur_score, ey
                for cand in candidates:
                    cand = max(min(cand, hi), lo)
                    cand = _round_to_step(cand, st)
                    if cand == base:
                        eyy, sc = ey, cur_score
                    else:
                        trial = dict(cur); trial[v] = cand
                        eyy = predict(trial)
                        sc = _objective(eyy, setpoint, trial)
                    if sc < best_score - 1e-9:
                        best_v, best_score, best_ey = cand, sc, eyy
                if best_v != base:
                    cur[v], cur_score, ey, improved = best_v, best_score, best_ey, True
            if not improved:
                break
        results.append({"adjustments": dict(cur), "expected_y": float(ey), "score": float(cur_score)})

    uniq: Dict[str, Dict[str, Any]] = {}
    for r in results:
        key = "|".join(f"{k}:{r['adjustments'][k]:.6f}" for k in variables)
        if key not in uniq or r["score"] < uniq[key]["score"]:
            uniq[key] = r
    ranked = sorted(uniq.values(), key=lambda x: x["score"])
    return ranked

def optimize_control(
    *,
    variables: List[str],
    last_y: float, delta_y: float,
    setpoint: float, horizon: int,
    constraints: Optional[Dict[str, Any]],
    top_k: int = 1,
    expected_y_fn = None,
    feature_df=None,
    target_col: Optional[str] = None,
    restarts: int = 12,
    iters_per_restart: int = 80,
    method: str = "coordinate",  # "coordinate" | "nelder-mead" | "powell" | "lbfgsb" | "tnc" | "slsqp" | "cobyla" | "trust-constr" | "de" | "anneal" | "auto"
    maxiter: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if random_seed is not None:
        random.seed(random_seed)
        try:
            _np.random.seed(random_seed)  # type: ignore
        except Exception:
            pass

    bounds = _make_bounds(variables, constraints)
    predict = _predict_wrapper(expected_y_fn, last_y, delta_y, horizon, feature_df, target_col)

    if method == "coordinate" or not _HAS_SCIPY:
        ranked = _coordinate_search(variables, bounds, predict, setpoint, restarts=restarts, iters_per_restart=iters_per_restart)
    else:
        var_idx = {v:i for i, v in enumerate(variables)}
        lo_vec = [_DEFAULT_BOUNDS[0]] * len(variables)
        hi_vec = [_DEFAULT_BOUNDS[1]] * len(variables)
        steps  = [_DEFAULT_STEPS] * len(variables)

        for i, v in enumerate(variables):
            lo, hi, st = bounds[v]
            lo_vec[i], hi_vec[i], steps[i] = lo, hi, st

        def vec_to_adj(x_vec):
            return {v: float(_round_to_step(max(min(x_vec[i], hi_vec[i]), lo_vec[i]), steps[i])) for i, v in enumerate(variables)}

        def f_obj(x_vec):
            adjs = vec_to_adj(x_vec)
            ey = predict(adjs)
            return _objective(ey, setpoint, adjs)

        x0 = _np.array([0.0 for _ in variables], dtype=float)

        ranked = []
        m = method.lower()
        try:
            if m in {"nelder-mead", "nelder", "nm"}:
                res = _spopt.minimize(f_obj, x0, method="Nelder-Mead", options={"maxiter": maxiter or 500})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"powell"}:
                res = _spopt.minimize(f_obj, x0, method="Powell", bounds=list(zip(lo_vec, hi_vec)), options={"maxiter": maxiter or 500})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"lbfgsb", "l-bfgs-b", "l-bfgs"}:
                res = _spopt.minimize(f_obj, x0, method="L-BFGS-B", bounds=list(zip(lo_vec, hi_vec)), options={"maxiter": maxiter or 300})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"tnc"}:
                res = _spopt.minimize(f_obj, x0, method="TNC", bounds=list(zip(lo_vec, hi_vec)), options={"maxiter": maxiter or 300})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"slsqp"}:
                res = _spopt.minimize(f_obj, x0, method="SLSQP", bounds=list(zip(lo_vec, hi_vec)), options={"maxiter": maxiter or 300})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"cobyla"}:
                # COBYLA는 bounds 직접 지원 X → 목적함수 내에서 clamp
                res = _spopt.minimize(f_obj, x0, method="COBYLA", options={"maxiter": maxiter or 500})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"trust-constr", "trust"}:
                res = _spopt.minimize(
                    f_obj, x0, method="trust-constr",
                    bounds=_spopt.Bounds(lo_vec, hi_vec),
                    options={"maxiter": maxiter or 300}
                )
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"de", "differential_evolution"}:
                res = _spopt.differential_evolution(
                    f_obj, bounds=list(zip(lo_vec, hi_vec)),
                    maxiter=maxiter or 1000, polish=True
                )
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"anneal", "dual_annealing"}:
                res = _spopt.dual_annealing(
                    f_obj, bounds=list(zip(lo_vec, hi_vec)),
                    maxiter=maxiter or 500
                )
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            elif m in {"auto"}:
                # 간단한 자동 전략: 전역 탐색(DE) → 국소(SLSQP)
                res_de = _spopt.differential_evolution(f_obj, bounds=list(zip(lo_vec, hi_vec)), maxiter=maxiter or 300, polish=False)
                res = _spopt.minimize(f_obj, res_de.x, method="SLSQP", bounds=list(zip(lo_vec, hi_vec)), options={"maxiter": 200})
                ranked = [{"adjustments": vec_to_adj(res.x), "expected_y": float(predict(vec_to_adj(res.x))), "score": float(res.fun)}]

            else:
                # 알 수 없는 메서드는 좌표탐색
                ranked = _coordinate_search(variables, bounds, predict, setpoint, restarts=restarts, iters_per_restart=iters_per_restart)

        except Exception:
            ranked = _coordinate_search(variables, bounds, predict, setpoint, restarts=restarts, iters_per_restart=iters_per_restart)

    out = []
    ranked = sorted(ranked, key=lambda x: x["score"])
    for i, r in enumerate(ranked[:top_k], start=1):
        out.append({
            "id": f"cand_{i}",
            "adjustments": r["adjustments"],
            "expected_y": round(float(r["expected_y"]), 6),
            "score": round(float(r["score"]), 6)
        })
    return out

from typing import Dict, Any, List, Optional, Tuple
import math, random

from utils import apply_constraints, predict_expected_y, score_candidate

try:
    import numpy as _np
    from scipy import optimize as _spopt 
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
_FREE = {
    "enabled": False,
    "no_bounds": True, 
    "no_round":  True, 
    "range":     5000.0, 
    "step":      0.1, 
}
_DEFAULT_BOUNDS = (-5, 5)
_DEFAULT_STEPS  = 1
def _method_rounding_policy(method_name: str) -> Dict[str, bool]:
    """
    각 SciPy 최적화기 특성에 맞춘 정책을 반환.
    - do_round:     스텝 격자 반올림 여부
    - do_clamp:     bounds 클램프를 vec_to_adj에서 수행할지 여부
    """
    m = (method_name or "").lower()
    # 연속계열: 라운딩 금지, clamp는 bounds가 지원되므로 불필요(=False)
    if m in {"lbfgsb", "l-bfgs-b", "l-bfgs", "tnc", "slsqp", "nelder-mead", "nelder", "nm", "powell", "trust-constr", "trustconstr"}:
        return {"do_round": False, "do_clamp": False}
    # COBYLA: bounds 미지원 → 반드시 clamp 필요. 라운딩은 끔.
    if m in {"cobyla"}:
        return {"do_round": False, "do_clamp": True}
    # 좌표탐색/그리드/격자 상정 메서드: 라운딩 유지, clamp도 수행(안전)
    if m in {"coordinate"}:
        return {"do_round": True, "do_clamp": True}
    # 그 외(예: DE/anneal 등): 기본은 라운딩 끔, clamp 수행
    return {"do_round": False, "do_clamp": True}
def _norm_step(lo: float, hi: float, step: Optional[float]) -> float:
    if step and step > 0:
        return float(step)
    span = max(hi - lo, 1e-6)
    return round(span / 3.0, 10)

def _round_to_step(x: float, step: float) -> float:
    # no_step(True)면 어떤 반올림도 하지 말 것 (정수 반올림 금지!)
    if _FREE["enabled"] and _FREE["no_round"]:
        return float(x)
    return float(round(round(x / step) * step, 10))

def _make_bounds(variables: List[str], constraints: Optional[Dict[str, Any]], feature_df=None) -> Dict[str, Tuple[float,float,float]]:
    """
    bounds와 step을 생성.
    - 사용자가 bounds[v].step을 명시하면 그대로 사용
    - 아니면 'step_policy'를 보고 스케일 인지형으로 step 자동 산정
        mode="percent_of_latest": step = max(pct * |latest|, min_abs)
        mode="std_of_feature":   step = max(std_mult * std(v), floor_frac * (hi-lo))
    - free_explore는 이전 동작 그대로 유지
    """
    _FREE["enabled"] = False
    if isinstance(constraints, dict) and constraints.get("free_explore"):
        fe = constraints["free_explore"]                      # ← 중첩 dict에서 읽기
        _FREE["enabled"]   = bool(fe.get("enabled", True))
        _FREE["no_bounds"] = bool(fe.get("no_bounds", True))
        _FREE["no_round"]  = bool(fe.get("no_step",  True))   # (키 이름 유지)
        _FREE["range"]     = float(fe.get("free_range", 50.0))
        _FREE["step"]      = float(fe.get("free_step",  0.1))

    bounds_cfg = (constraints or {}).get("bounds", {}) if isinstance(constraints, dict) else {}

    # 스케일 인지형 step 정책 (기본값: percent_of_latest)
    sp = (constraints or {}).get("step_policy", {}) if isinstance(constraints, dict) else {}
    mode        = str(sp.get("mode", "percent_of_latest")).lower()
    pct         = float(sp.get("pct", 0.01))         # 1% of latest
    min_abs     = float(sp.get("min_abs", 0.1))      # 최소 step
    std_mult    = float(sp.get("std_mult", 0.2))     # 표준편차 기반일 때 배수
    floor_frac  = float(sp.get("floor_frac", 0.02))  # (hi-lo) 대비 하한

    out: Dict[str, Tuple[float,float,float]] = {}
    for v in variables:
        if _FREE["enabled"] and _FREE["no_bounds"]:
            lo, hi = -_FREE["range"], _FREE["range"]
            # step: free_explore에서 no_round=True면 사실상 무시되지만 일단 기록
            st = _FREE["step"]
            out[v] = (lo, hi, float(st))
            continue

        # 1) 경계 결정
        b  = bounds_cfg.get(v, {})
        lo = float(b.get("min", _DEFAULT_BOUNDS[0]))
        hi = float(b.get("max", _DEFAULT_BOUNDS[1]))

        # 2) step 우선순위
        if "step" in b and float(b.get("step", 0)) > 0:
            st = float(b["step"])  # 사용자가 명시한 step을 그대로 사용
        else:
            # 스케일 인지형 자동 step
            st = None
            if mode == "percent_of_latest" and feature_df is not None and v in feature_df.columns:
                latest_val = float(feature_df.iloc[-1][v])
                st = max(abs(latest_val) * pct, min_abs)
                # 너무 작은 hi-lo 구간을 대비해 하한을 약간 둠
                st = max(st, abs(hi - lo) * floor_frac)
            elif mode == "std_of_feature" and feature_df is not None and v in feature_df.columns:
                try:
                    col = feature_df[v].astype(float).values
                    stdv = float(np.nanstd(col))  # type: ignore
                except Exception:
                    stdv = 0.0
                st = max(stdv * std_mult, abs(hi - lo) * floor_frac, min_abs)
            else:
                # 폴백: 기존 규칙(span/3)
                st = _norm_step(lo, hi, None)

        out[v] = (lo, hi, round(float(st), 10))
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
    restarts=12, iters_per_restart=80,
    debug: bool = True,
    plateau_expand: bool = True,
):
    def clamp_round_vec(adjs: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x_in = float(adjs.get(v, 0.0))
            if not (_FREE["enabled"] and _FREE["no_bounds"]):
                x_in = max(min(x_in, hi), lo)
            out[v] = _round_to_step(x_in, st)
        return out

    def random_init() -> Dict[str, float]:
        adjs = {}
        for v in variables:
            lo, hi, st = bounds[v]
            x = random.uniform(lo, hi)
            adjs[v] = _round_to_step(x, st)
        return adjs

    def try_candidates(cur, v, base, st, cur_score, ey, scale=1.0):
        lo, hi, _ = bounds[v]
        s = st * scale
        cands = [base, base + s, base - s]
        best_v, best_score, best_ey = base, cur_score, ey
        all_equal = True
        rows = []
        for cand in cands:
            if not (_FREE["enabled"] and _FREE["no_bounds"]):
                cand = max(min(cand, hi), lo)
            cand = _round_to_step(cand, st)
            if cand == base:
                eyy, sc = ey, cur_score
            else:
                trial = dict(cur); trial[v] = cand
                eyy = predict(trial)
                sc  = _objective(eyy, setpoint, trial)
            rows.append((cand, eyy, sc))
            if abs(sc - cur_score) > 1e-12:
                all_equal = False
            if sc < best_score - 1e-9:
                best_v, best_score, best_ey = cand, sc, eyy
        return best_v, best_score, best_ey, all_equal, rows

    results: List[Dict[str, Any]] = []
    for ri in range(restarts):
        if ri == 0:
            cur = clamp_round_vec({v: 0.0 for v in variables})
        else:
            cur = random_init()
        ey = predict(cur)
        cur_score = _objective(ey, setpoint, cur)
        if debug:
            print(f"[restart {ri}] init cur={cur} ey={ey:.6f} score={cur_score:.6f}")

        for _iter in range(iters_per_restart):
            improved = False
            if debug:
                print(f"  [iter {_iter}]")
            for v in variables:
                lo, hi, st = bounds[v]
                base = cur[v]
                best_v, best_score, best_ey, all_equal, rows = try_candidates(cur, v, base, st, cur_score, ey, 1.0)
                if debug:
                    print(f"    var={v} base={base:.6f} step={st:.6f} rows={[(round(c,6), round(e,6), round(s,6)) for c,e,s in rows]}")

                # plateau면 step을 키우며 분기선을 넘겨봄
                if plateau_expand and all_equal:
                    scale = 2.0
                    while scale <= 32.0:
                        bv, bs, be, ae, rows2 = try_candidates(cur, v, base, st, cur_score, ey, scale)
                        if debug:
                            print(f"      plateau→ scale x{scale:.1f} rows={[(round(c,6), round(e,6), round(s,6)) for c,e,s in rows2]}")
                        if (bv != base) or (not ae) or (bs < best_score - 1e-9):
                            best_v, best_score, best_ey = bv, bs, be
                            break
                        scale *= 2.0

                if best_v != base:
                    if debug:
                        print(f"      update {v}: {base:.6f} → {best_v:.6f} | score {cur_score:.6f} → {best_score:.6f}")
                    cur[v], cur_score, ey, improved = best_v, best_score, best_ey, True

            if not improved:
                if debug:
                    print(f"  no improvement; stop iter at { _iter }")
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
    method: str = "coordinate",
    maxiter: Optional[int] = None,
    random_seed: Optional[int] = None,
    debug: bool = True,               # <-- ADD
    plateau_expand: bool = True,      # <-- ADD
) -> List[Dict[str, Any]]:
    if random_seed is not None:
        random.seed(random_seed)
        try:
            _np.random.seed(random_seed)  # type: ignore
        except Exception:
            pass
    
    bounds = _make_bounds(variables, constraints, feature_df=feature_df)
    predict = _predict_wrapper(expected_y_fn, last_y, delta_y, horizon, feature_df, target_col)
    
    print(">>> variables =", variables)
    print(">>> bounds =", bounds)
    print(">>> setpoint/horizon/last_y/delta_y =", setpoint, horizon, last_y, delta_y)

    # === ADD: sensitivity report (각 변수 한 칸(step) 움직였을 때 예측/스코어 변화) ===
    if debug:
        base = {v: 0.0 for v in variables}
        y0 = predict(base)
        s0 = _objective(y0, setpoint, base)
        print(">>> sensitivity (per step):")
        for v in variables:
            st = bounds[v][2]
            for dv in (st, -st):
                trial = dict(base); trial[v] = trial[v] + dv
                y1 = predict(trial)
                s1 = _objective(y1, setpoint, trial)
                print(f"    {v} {dv:+.6f} → Δy={y1 - y0:+.6f}, Δscore={s1 - s0:+.6f}")
    if method == "coordinate" or not _HAS_SCIPY:
        ranked = _coordinate_search(
            variables, bounds, predict, setpoint,
            restarts=restarts, iters_per_restart=iters_per_restart,
            debug=debug, plateau_expand=plateau_expand
        )
    else:
        var_idx = {v:i for i, v in enumerate(variables)}
        lo_vec = [_DEFAULT_BOUNDS[0]] * len(variables)
        hi_vec = [_DEFAULT_BOUNDS[1]] * len(variables)
        steps  = [_DEFAULT_STEPS] * len(variables)
        policy = _method_rounding_policy(method)
        _do_round = policy["do_round"]
        _do_clamp = policy["do_clamp"]

        print(f"[OPT] method={method!r} policy: round={'ON' if _do_round else 'OFF'}, clamp={'ON' if _do_clamp else 'OFF'}")

        def _maybe_round(x: float, st: float) -> float:
            # free_explore.no_step=True 이면 강제로 라운딩 금지
            if _FREE["enabled"] and _FREE["no_round"]:
                return float(x)
            return float(_round_to_step(x, st)) if _do_round else float(x)
        for i, v in enumerate(variables):
            lo, hi, st = bounds[v]
            lo_vec[i], hi_vec[i], steps[i] = lo, hi, st

        def vec_to_adj(x_vec):
            out = {}
            for i, v in enumerate(variables):
                x = float(x_vec[i])
                # clamp: COBYLA 또는 좌표/기타 안전성 위해 정책상 True일 때
                if _do_clamp and not (_FREE["enabled"] and _FREE["no_bounds"]):
                    x = max(min(x, hi_vec[i]), lo_vec[i])
                # round-to-step: 메서드 정책과 free_explore.no_step에 따름
                out[v] = _maybe_round(x, steps[i])
            return out

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
    print(">>> top candidates (score, expected_y, adjustments):")
    for i, r in enumerate(ranked[:top_k], start=1):
        print(f"  cand_{i}: score={r['score']:.6f}, expected_y={r['expected_y']:.6f}, adj={r['adjustments']}")
        out.append({
            "id": f"cand_{i}",
            "adjustments": r["adjustments"],
            "expected_y": round(float(r["expected_y"]), 6),
            "score": round(float(r["score"]), 6)
        })
    return out

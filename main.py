# main.py — AutoControl API (prediction agent 연동 + surrogate/optimizer 확장)
import os, json, uuid, pathlib, logging, base64, hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv

from autocontrol.schema import (
    AutoControlRunRequest, AutoControlRunResponse,
    AutoControlRunResult, ControlCandidate,
    AutoControlSpec
)
from orchestra.schema import (
    AgentAssignment, OrchestrationAssignRequest,
    UpdatedAssignment, OrchestrationAssignResponse, AgentExecutionOrder
)
from utils import ensure_dirs, now_iso
from optimizer import optimize_control
from surrogate import load_surrogate, train_and_save_linear_simple

try:
    from surrogate import train_and_save_sklearn 
    HAS_SKLEARN_TRAIN = True
except Exception:
    HAS_SKLEARN_TRAIN = False

from module import risk_module, explain_module
from llm_io import LLMBridge

import requests 

load_dotenv()
API_PORT = int(os.getenv("PORT", "8014"))

BASE_DIR          = pathlib.Path(__file__).resolve().parent
DATA_DIR_DEFAULT  = (BASE_DIR / "autocontrol" / "data" / "Industrial_DB_sample").resolve()
DEFAULT_CSV       = (DATA_DIR_DEFAULT / "SEMI_CMP_SENSORS.csv").resolve()
OUTPUTS_DIR       = (BASE_DIR / "outputs").resolve()
AUTOCONTROL_DIR   = (OUTPUTS_DIR / "autocontrol").resolve()
WEIGHTS_CACHE_DIR = (OUTPUTS_DIR / "weights").resolve()
ensure_dirs([OUTPUTS_DIR, AUTOCONTROL_DIR, WEIGHTS_CACHE_DIR])

def _safe_filename(prefix: str, ext: str) -> pathlib.Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = uuid.uuid4().hex[:8]
    return (WEIGHTS_CACHE_DIR / f"{prefix}_{ts}_{rid}{ext}").resolve()

def _save_base64_to_file(b64: str, preferred_ext: str = ".npz") -> str:
    raw = base64.b64decode(b64.encode("utf-8"))
    ext = preferred_ext
    head4 = raw[:4]
    if head4 == b"\x80\x04\x95\x00":  
        ext = ".pkl"
    out = _safe_filename("weights_b64", ext)
    with open(out, "wb") as f:
        f.write(raw)
    return str(out)

def _looks_supported_weight(path: str) -> bool:
    low = path.lower()
    return low.endswith(".npz") or low.endswith(".joblib") or low.endswith(".pkl")

app = FastAPI(
    title="AutoControl API",
    description="Surrogate + Optimizer",
    version="1.0.0"
)


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/docs")

def fetch_agent_snapshot(base_url: str, task_id: str, inline_base64: bool = True) -> dict:
    """
    다른 에이전트의 스냅샷 엔드포인트 호출:
      GET {base_url}/api/v1/prediction/activate_autocontrol/{taskId}?inline_base64=true|false
    성공 시 응답의 'data' 딕셔너리를 그대로 반환.
    실패 시 예외를 던짐 (requests.HTTPError 등).
    """
    url = f"{str(base_url).rstrip('/')}/api/v1/prediction/activate_autocontrol/{task_id}"
    resp = requests.get(url, params={"inline_base64": str(inline_base64).lower()}, timeout=15)
    resp.raise_for_status()
    try:
        j = resp.json()
    except Exception as e:
        raise RuntimeError(f"invalid JSON from prediction agent: {e} | text={resp.text[:200]}")
    if not isinstance(j, dict) or j.get("code") != "SUCCESS" or "data" not in j:
        raise RuntimeError(f"unexpected snapshot response: {j}")
    return j["data"]  # keys: taskId, best_model, feature_names, target_col, prediction, weight_* ...

def realize_weight_from_snapshot(snap: dict) -> str | None:
    """
    스냅샷에서 base64 인라인 weight가 있으면 로컬 파일로 저장하고 경로 반환.
    - 저장 위치 우선순위: WEIGHTS_CACHE_DIR → AUTOCONTROL_DIR → /tmp
    - weight_sha256 있으면 무결성 검증
    - 인라인이 없거나 실패하면 None
    """
    inlined = bool(snap.get("weight_inlined"))
    b64     = snap.get("weight_base64")
    fname0  = snap.get("weight_filename")
    sha_ref = snap.get("weight_sha256")

    if not (inlined and b64):
        # 인라인이 아닌 경우(상대 서버 로컬 경로만 있는 경우)는 여기서 처리하지 않음
        return None

    # 저장 디렉토리 선택
    def _pick_save_dir() -> Path:
        for name in ("WEIGHTS_CACHE_DIR", "AUTOCONTROL_DIR"):
            p = globals().get(name)
            try:
                if p:
                    return Path(str(p))
            except Exception:
                pass
        return Path("/tmp")

    # 확장자 유지
    ext = ""
    if isinstance(fname0, str) and "." in fname0:
        ext = "." + fname0.rsplit(".", 1)[-1]
    if not ext:
        ext = ".bin"

    save_dir = _pick_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_fp = save_dir / f"agent_{snap.get('taskId','unknown')}{ext}"

    # base64 decode → 저장
    try:
        blob = base64.b64decode(b64.encode("ascii"))
    except Exception as e:
        raise RuntimeError(f"base64 decode failed: {e}")
    with open(out_fp, "wb") as f:
        f.write(blob)

    # sha256 검증(가능하면)
    if sha_ref:
        h = hashlib.sha256()
        with open(out_fp, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        sha_now = h.hexdigest()
        if sha_now.lower() != str(sha_ref).lower():
            try:
                out_fp.unlink(missing_ok=True)
            finally:
                raise RuntimeError(f"sha256 mismatch: got={sha_now} ref={sha_ref}")

    return str(out_fp)

def _compose_short_answer_ko(target_col: str, setpoint: float, horizon: int, top: Dict[str, Any]) -> str:
    adjs = top.get("adjustments") or {}
    if adjs:
        adj_txt = ", ".join(f"{k} {v:+.3f}" for k, v in adjs.items()) + " 로 조정하세요"
    else:
        adj_txt = "추가 조정 없이 현재 설정을 유지하세요"
    ey = float(top.get("expected_y", 0.0))
    sc = float(top.get("score", 0.0))
    return (
        f"{horizon} 스텝 후 '{target_col}'을(를) {setpoint:.3f}에 맞추려면 {adj_txt}. "
        f"예상값 {ey:.3f}, 점수 {sc:.3f}."
    )
@app.post("/api/v1/autocontrol/run-direct", response_model=AutoControlRunResponse)
def ac_run_direct(
    body: AutoControlRunRequest = Body(...),

    surrogate_model: str = Query(
        "linear",
        description="'linear' 또는 'sklearn:<registry_key>' (예: 'sklearn:random_forest_regressor')"
    ),
    surrogate_params: Optional[str] = Query(
        None,
        description="하이퍼파라미터(JSON 문자열). 예: '{\"n_estimators\":200}'"
    ),
    opt_method: str = Query(
        "coordinate",
        description="SciPy 기반 메서드 선택. 예 : coordinate | nelder-mead | powell | lbfgsb | tnc | slsqp | cobyla | trust-constr | de | anneal | auto"
    ),
    maxiter: Optional[int] = Query(None, description="SciPy 기반 메서드의 최대 반복 수"),
    agent_base_url: Optional[str] = Query(
        None, description="Prediction Agent 베이스 URL (예: http://localhost:8001)"
    ),
    agent_task_id: Optional[str] = Query(
        None, description="Orchestration Agent로부터 받은 taskId"
    ),
    agent_inline: bool = Query(
        True, description="base64 인라인 요청 여부"
    ),
    weight_local_path: Optional[str] = Query(None),
):
    # --- Query wrapper to plain value (FastAPI Query → value) -------------------
    try:
        from fastapi.params import Query as _Q
        def _q2val(x, typ=None):
            if isinstance(x, _Q):
                x = x.default
            if typ and x is not None and not isinstance(x, typ):
                return None
            return x
        weight_local_path = _q2val(weight_local_path, (str, bytes, os.PathLike))
        agent_base_url    = _q2val(agent_base_url, (str,))
        agent_task_id     = _q2val(agent_task_id, (str,))
        if isinstance(agent_inline, _Q):  # bool 그대로
            agent_inline = bool(agent_inline.default)
    except Exception:
        pass
    surrogate_model = (surrogate_model or "linear").strip()
    opt_method      = (opt_method or "coordinate").strip()
    events: List[str] = []

    # --- Load CSV ----------------------------------------------------------------
    csv_path = body.csv_path and getattr(body.csv_path, "csv_path", None)
    if not csv_path:
        csv_path = str(DEFAULT_CSV)
    if not pathlib.Path(csv_path).exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {csv_path}")
    feature_df = pd.read_csv(csv_path)

    if body.target_col not in feature_df.columns:
        raise HTTPException(status_code=400, detail=f"target_col '{body.target_col}' not in CSV columns")

    feature_names = body.feature_names or [c for c in feature_df.columns if c != body.target_col]

    try:
        last_y = float(feature_df[body.target_col].iloc[-1])
        if len(feature_df) >= 2:
            prev_y = float(feature_df[body.target_col].iloc[-2])
            delta_y = last_y - prev_y
        else:
            delta_y = 0.0
    except Exception:
        last_y, delta_y = 0.0, 0.0

    # ===================== 1) Prediction Agent → weight ==========================
    # 의도: Prediction Agent에서 가능한 경우 weight를 받아 로컬 파일 경로로 확정
    realized = None
    if agent_base_url and agent_task_id:
        try:
            agent_snapshot = fetch_agent_snapshot(agent_base_url, agent_task_id, inline_base64=agent_inline)
            events.append(f"[{now_iso()}] fetched snapshot from prediction agent: taskId={agent_task_id}")
            realized = realize_weight_from_snapshot(agent_snapshot)
            if realized:
                weight_local_path = realized
                events.append(f"[{now_iso()}] weights realized from agent snapshot → {os.path.basename(realized)}")
            else:
                events.append(f"[{now_iso()}] agent snapshot had no usable weights; will try local/fallback")
        except HTTPException as e:
            raise e
        except Exception as e:
            events.append(f"[{now_iso()}] agent snapshot fetch error: {e}; will try local/fallback")

    # ===================== 2) local weight path 사용 (있다면) =====================
    # - 제공된 weight_local_path가 파일로 '존재'하면 그대로 사용
    # - 제공되었지만 '없다면', 3단계 학습 후 그 경로에 저장 시도
    want_save_to_path: Optional[pathlib.Path] = None
    if not realized and weight_local_path:
        p = pathlib.Path(str(weight_local_path))
        if p.exists() and p.is_file():
            events.append(f"[{now_iso()}] using provided local weight: {p.name}")
            # weight_local_path 그대로 유지
        else:
            # 지정된 경로가 파일로 존재하지 않음 → 학습 후 여기에 저장
            want_save_to_path = p
            events.append(f"[{now_iso()}] desired weight path not found → will train and save to: {p}")

    # ===================== 3) 없으면 surrogate 학습하여 저장 ======================
    if not weight_local_path or (want_save_to_path is not None and not want_save_to_path.exists()):
        # 학습이 필요한 조건:
        # - weight_local_path가 비어있다, 또는
        # - 사용자가 원하는 저장 경로(want_save_to_path)가 있는데 아직 파일이 없다
        import shutil

        if isinstance(surrogate_model, str) and surrogate_model.startswith("sklearn:"):
            if not HAS_SKLEARN_TRAIN:
                raise HTTPException(
                    status_code=400,
                    detail="surrogate.py가 scikit-learn 학습을 지원하지 않습니다. (확장판으로 교체 필요)"
                )
            sk_key = surrogate_model.split(":", 1)[1].strip()
            params = {}
            if surrogate_params:
                try:
                    params = json.loads(surrogate_params)
                except Exception as e:
                    events.append(f"[{now_iso()}] surrogate_params JSON parse failed: {e}")
            ckpt = train_and_save_sklearn(   # type: ignore
                feature_df=feature_df,
                target_col=body.target_col,
                save_dir=WEIGHTS_CACHE_DIR,
                x_cols=feature_names,
                model={"name": sk_key, "params": params},
                filename_stub=f"{body.acID or uuid.uuid4().hex[:6]}_{sk_key}"
            )
            trained_path = pathlib.Path(str(ckpt))
            events.append(f"[{now_iso()}] trained sklearn surrogate: {sk_key} → {trained_path.name}")
        else:
            ckpt = train_and_save_linear_simple(
                feature_df=feature_df,
                target_col=body.target_col,
                save_dir=WEIGHTS_CACHE_DIR,
                x_cols=feature_names,
                filename_stub=f"{body.acID or uuid.uuid4().hex[:6]}_linear"
            )
            trained_path = pathlib.Path(str(ckpt))
            events.append(f"[{now_iso()}] trained linear surrogate → {trained_path.name}")

        # 학습 결과를 최종 weight_local_path로 확정
        if want_save_to_path:
            try:
                want_save_to_path.parent.mkdir(parents=True, exist_ok=True)
                # 사용자가 지정한 파일명으로 이동(확실히 저장 경로를 의도대로)
                shutil.move(str(trained_path), str(want_save_to_path))
                weight_local_path = str(want_save_to_path)
                events.append(f"[{now_iso()}] moved trained weight → {want_save_to_path.name}")
            except Exception as e:
                # 이동 실패 시: 학습된 경로 그대로 사용
                weight_local_path = str(trained_path)
                events.append(f"[{now_iso()}] failed to move weight to desired path ({e}); using {trained_path.name}")
        else:
            # 사용자가 별도 경로를 요구하지 않았다면 학습 경로 그대로 사용
            weight_local_path = str(trained_path)

    # ===================== Surrogate 로딩 및 최적화 ================================
    expected_y_fn = load_surrogate(weight_local_path)

    setpoint = float(body.control.setpoint)
    horizon  = int(body.control.horizon)
    ranked = optimize_control(
        variables=feature_names,
        last_y=last_y,
        delta_y=delta_y,
        setpoint=setpoint,
        horizon=horizon,
        constraints=body.constraints,
        top_k=5,
        expected_y_fn=expected_y_fn,
        feature_df=feature_df,
        target_col=body.target_col,
        method=opt_method,
        maxiter=maxiter
    )

    for r in ranked:
        r["setpoint_hint"] = setpoint
    risk = risk_module(feature_df, body.target_col, ranked)
    explain = explain_module(feature_df, body.target_col, ranked)

    # ===================== 결과 파일/응답 구성 ====================================
    task_id = body.taskId
    ac_id   = body.acID
    out_name = f"ac_{task_id}_{ac_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    out_csv  = (AUTOCONTROL_DIR / out_name).resolve()
    rows = []
    adj_cols = sorted({k for r in ranked for k in (r.get("adjustments") or {}).keys()})
    for r in ranked:
        base = {"id": r["id"], "expected_y": r["expected_y"], "score": r["score"]}
        for c in adj_cols:
            base[c] = (r["adjustments"] or {}).get(c, 0.0)
        rows.append(base)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if ranked:
        nl_answer = _compose_short_answer_ko(body.target_col, setpoint, horizon, ranked[0])
    else:
        nl_answer = "추천 조정안을 찾지 못했습니다."
    events.append(f"[{now_iso()}] NL_ANSWER: {nl_answer}")

    ans_path = out_csv.with_suffix(".txt")
    with open(ans_path, "w", encoding="utf-8") as f:
        f.write(nl_answer + "\n")
    events.append(f"[{now_iso()}] natural-language answer saved → {ans_path.name}")
    events.append(f"[{now_iso()}] optimized with method={opt_method}, maxiter={maxiter}")
    events.append(f"[{now_iso()}] result csv saved → {out_csv.name}")

    selected_idx = 0 if ranked else -1
    candidates: List[ControlCandidate] = [
        ControlCandidate(**{
            "id": r["id"],
            "adjustments": r["adjustments"],
            "expected_y": float(r["expected_y"]),
            "score": float(r["score"]),
        }) for r in ranked
    ]

    data = AutoControlRunResult(
        taskId=task_id,
        acID=ac_id,
        input_csv_path=str(csv_path),
        result_csv_path=str(out_csv),
        feature_names=feature_names,
        target_col=body.target_col,
        modelSelected=surrogate_model,
        weight_path=weight_local_path,
        control=body.control,
        candidates=candidates,
        selected_candidate_idx=selected_idx,
        risk=risk,
        explanation=explain,
        events=events
    )

    # (중복 계산 방지 위해 위에서 만든 nl_answer 그대로 사용)
    meta = {
        "timestamp": now_iso(),
        "surrogate_model": surrogate_model,
        "opt_method": opt_method,
        "maxiter": str(maxiter),
    }
    narration = None
    try:
        bridge = LLMBridge()
        narration = bridge._narrate({
            "nl_query": body.model_dump(),
            "results": [c.model_dump() for c in candidates],
            "risk": risk,
            "explain": explain,
        })
    except Exception as e:
        narration = f"[자동 설명 실패] {e}"

    return AutoControlRunResponse(
        code="SUCCESS",
        data=data,
        metadata={**meta, "narration": narration, "nl_answer": nl_answer}
    )



@app.put("/api/v1/task/{task_id}/autocontrol/assign", response_model=OrchestrationAssignResponse)
def orchestration_assign(task_id: str, req: OrchestrationAssignRequest = Body(...)):

    target = None
    for a in (req.agent_assignments or []):
        if (a.agent_id or "").lower() == "autocontrol":
            target = a
            break
    if not target:
        return OrchestrationAssignResponse(
            task_id=req.task_id or task_id,
            updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="assigned")]
        )

    nl_query, opts = None, {}
    eo = target.execution_order
    if isinstance(eo, dict) and "nl_query" in eo:
        nl_query = str(eo.get("nl_query") or "").strip()
        opts = eo.get("options") or {}
    elif isinstance(eo, str):
        nl_query = eo.strip()
    elif isinstance(eo, AgentExecutionOrder):
        nl_query = getattr(eo, "nl_query", None)
        opts = getattr(eo, "options", {})
    else:
        return OrchestrationAssignResponse(
            task_id=req.task_id or task_id,
            updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="assigned")]
        )
    if not nl_query:
        return OrchestrationAssignResponse(
            task_id=req.task_id or task_id,
            updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="assigned")]
        )

    bridge = LLMBridge()
    spec = bridge._extract_autocontrol_spec(nl_query)

    constraints = spec.get("constraints") or {}
    if isinstance(opts, dict) and opts:
        constraints.update(opts)
        if isinstance(opts.get("feature_names"), list):
            spec["feature_names"] = opts["feature_names"]
    spec["constraints"] = constraints

    ctrl = spec.get("control") or {}
    setpoint = spec.get("setpoint", ctrl.get("setpoint"))
    horizon  = spec.get("horizon",  ctrl.get("horizon", 11))

    if setpoint is None:
        raise HTTPException(status_code=400, detail="setpoint missing from spec/control")
    try:
        setpoint = float(setpoint)
        horizon  = int(horizon)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid setpoint/horizon type")

    body = AutoControlRunRequest(
        taskId=spec.get("taskId"),
        acID=spec.get("acID"),
        feature_names=spec.get("feature_names") or [],
        target_col=spec["target_col"],
        control={"setpoint": setpoint, "horizon": horizon},
        constraints=spec.get("constraints"),
    )
    surrogate_model = getattr(eo, "surrogate_model", None) or eo.get("surrogate_model") if isinstance(eo, dict) else None
    surrogate_params = getattr(eo, "surrogate_params", None) or eo.get("surrogate_params") if isinstance(eo, dict) else None
    opt_method = getattr(eo, "opt_method", None) or eo.get("opt_method") if isinstance(eo, dict) else None
    maxiter = getattr(eo, "maxiter", None) or eo.get("maxiter") if isinstance(eo, dict) else None
    agent_base_url = getattr(eo, "agent_base_url", None) or eo.get("agent_base_url") if isinstance(eo, dict) else None
    agent_task_id = getattr(eo, "agent_task_id", None) or eo.get("agent_task_id") if isinstance(eo, dict) else None
    agent_inline = getattr(eo, "agent_inline", True) if hasattr(eo, "agent_inline") else eo.get("agent_inline", True) if isinstance(eo, dict) else True
    weight_local_path = getattr(eo, "weight_local_path", None) or eo.get("weight_local_path") if isinstance(eo, dict) else None
    res = ac_run_direct(
        body,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        opt_method=opt_method,
        maxiter=maxiter,
        agent_base_url=agent_base_url,
        agent_task_id=agent_task_id,
        agent_inline=agent_inline,
        weight_local_path=weight_local_path,
    )

    nl_answer = None
    try:
        nl_answer = (res.metadata or {}).get("nl_answer")
    except Exception:
        nl_answer = None
    if not nl_answer:
        top = None
        if getattr(res.data, "candidates", None):
            idx = getattr(res.data, "selected_candidate_idx", 0) or 0
            cand = res.data.candidates[idx]
            top = cand.model_dump() if hasattr(cand, "model_dump") else {
                "adjustments": getattr(cand, "adjustments", {}),
                "expected_y": getattr(cand, "expected_y", 0.0),
                "score": getattr(cand, "score", 0.0),
            }
        nl_answer = _compose_short_answer_ko(
            res.data.target_col, float(res.data.control.setpoint), int(res.data.control.horizon), top or {}
        )

    payload = OrchestrationAssignResponse(
        task_id=req.task_id or task_id,
        updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="ready")],
        response={
            "autocontrol": {
                "nl_answer": nl_answer,
                "result_csv_path": res.data.result_csv_path,
                "selected_candidate": (
                    res.data.candidates[res.data.selected_candidate_idx].model_dump()
                    if res.data.candidates else None
                ),
            }
        }
    )
    return payload

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)

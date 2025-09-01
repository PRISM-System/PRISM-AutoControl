# main.py — AutoControl API (prediction agent 연동 + surrogate/optimizer 확장)
import os, json, uuid, pathlib, logging, base64
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from autocontrol.schema import (
    AutoControlRunRequest, AutoControlRunResponse,
    AutoControlRunResult, ControlCandidate,
    AC_NLRequest, AC_NLParsedResponse,
    AutoControlSpec, NarrateRequest, NarrateResponse
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
    return """<h3>AutoControl API</h3>
    <p>POST <code>/api/v1/autocontrol/run-direct</code></p>
    <p>POST <code>/api/v1/autocontrol/parse-nl</code> / <code>/api/v1/autocontrol/narrate</code></p>
    """

def fetch_agent_snapshot(
    base_url: str,
    task_id: str,
    inline_base64: bool = True
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/prediction/activate_autocontrol/{task_id}"
    params = {"inline_base64": "true" if inline_base64 else "false"}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"agent responded {r.status_code}: {r.text[:200]}")
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"agent snapshot fetch failed: {e}")

def realize_weight_from_snapshot(snap: Dict[str, Any]) -> Optional[str]:
    data = snap.get("data") or {}
    b64 = data.get("weight_base64")
    path = data.get("weight_path")

    if b64:
        try:
            save_path = _save_base64_to_file(b64)
            return save_path if _looks_supported_weight(save_path) else None
        except Exception:
            pass

    if path and os.path.exists(path):
        return path if _looks_supported_weight(path) else None

    return None

# ──────────────────────────────────────────────────────────────────────
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
    weight_path: Optional[str] = Query(None, description="로컬 weight 경로(.npz/.joblib/.pkl)"),
    weight_url: Optional[str]  = Query(None, description="weight 파일 URL"),
    weights_b64: Optional[str] = Query(None, description="Base64 인코딩된 weight"),

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
        None, description="Prediction Agent의 taskId"
    ),
    agent_inline: bool = Query(
        True, description="base64 인라인 요청 여부"
    ),
):
    events: List[str] = []

    csv_path = body.control and getattr(body.control, "csv_path", None)
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

    weight_local_path: Optional[str] = None

    if weights_b64:
        try:
            weight_local_path = _save_base64_to_file(weights_b64)
            events.append(f"[{now_iso()}] weights: received via base64 → {os.path.basename(weight_local_path)}")
        except Exception as e:
            events.append(f"[{now_iso()}] base64 decode failed: {e}")

    if not weight_local_path and weight_url:
        try:
            r = requests.get(weight_url, timeout=20)
            r.raise_for_status()
            ext = ".npz"
            low = weight_url.lower()
            if low.endswith(".joblib"):
                ext = ".joblib"
            elif low.endswith(".pkl"):
                ext = ".pkl"
            out = _safe_filename("weights_url", ext)
            with open(out, "wb") as f:
                f.write(r.content)
            weight_local_path = str(out)
            events.append(f"[{now_iso()}] weights: downloaded from url → {os.path.basename(out)}")
        except Exception as e:
            events.append(f"[{now_iso()}] weight_url fetch failed: {e}")

    if not weight_local_path and weight_path and os.path.exists(weight_path):
        weight_local_path = weight_path
        events.append(f"[{now_iso()}] weights: using local path → {os.path.basename(weight_path)}")

    agent_snapshot = None
    if not weight_local_path and agent_base_url and agent_task_id:
        try:
            agent_snapshot = fetch_agent_snapshot(agent_base_url, agent_task_id, inline_base64=agent_inline)
            events.append(f"[{now_iso()}] fetched snapshot from prediction agent: taskId={agent_task_id}")
            realized = realize_weight_from_snapshot(agent_snapshot)
            if realized:
                weight_local_path = realized
                events.append(f"[{now_iso()}] weights: realized from agent snapshot → {os.path.basename(realized)}")
            else:
                events.append(f"[{now_iso()}] agent snapshot provided unsupported weight format; will train local surrogate")
        except HTTPException as e:
            raise e
        except Exception as e:
            events.append(f"[{now_iso()}] agent snapshot fetch error: {e}")

    if not weight_local_path:
        if surrogate_model.startswith("sklearn:"):
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
            weight_local_path = str(ckpt)
            events.append(f"[{now_iso()}] trained sklearn surrogate: {sk_key} → {os.path.basename(weight_local_path)}")
        else:
            ckpt = train_and_save_linear_simple(
                feature_df=feature_df,
                target_col=body.target_col,
                save_dir=WEIGHTS_CACHE_DIR,
                x_cols=feature_names,
                filename_stub=f"{body.acID or uuid.uuid4().hex[:6]}_linear"
            )
            weight_local_path = str(ckpt)
            events.append(f"[{now_iso()}] trained linear surrogate → {os.path.basename(weight_local_path)}")

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
    nl_answer = _compose_short_answer_ko(body.target_col, setpoint, horizon, ranked[0]) if ranked else "추천 후보를 찾지 못했습니다."
    events.append(f"[{now_iso()}] natural-language answer composed")
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
            "query": body.model_dump(), 
            "results": [c.model_dump() for c in candidates],  
            "risk": risk,
            "explain": explain,
        })
    except Exception as e:
        narration = f"[자동 설명 실패] {e}"

    return AutoControlRunResponse(
        code="SUCCESS",
        data=data,
        metadata={**meta, "narration": narration, "nl_answer": nl_answer}  # narration 추가
    )

@app.post("/api/v1/autocontrol/parse-nl", response_model=AC_NLParsedResponse)
def parse_nl(req: AC_NLRequest = Body(...)):
    bridge = LLMBridge()
    spec_dict = bridge._extract_autocontrol_spec(req.query)
    spec_dict.setdefault("feature_names", [])
    spec_dict.setdefault("constraints", None)
    ctrl = {
        "setpoint": float(spec_dict.get("setpoint")),
        "horizon": int(spec_dict.get("horizon", 11)),
    }
    ac_spec = AutoControlSpec(
        taskId=spec_dict.get("taskId"),
        acID=spec_dict.get("acID"),
        feature_names=spec_dict.get("feature_names"),
        target_col=spec_dict.get("target_col"),
        control=ctrl,
        constraints=spec_dict.get("constraints"),
    )
    return AC_NLParsedResponse(
        code="SUCCESS",
        data=ac_spec,
        metadata={"parsedBy": "LLMBridge._extract_autocontrol_spec"}
    )

@app.post("/api/v1/autocontrol/narrate", response_model=NarrateResponse)
def narrate(req: NarrateRequest = Body(...)):
    bridge = LLMBridge()
    try:
        narration = bridge._narrate(req.payload)
        return NarrateResponse(
            code="SUCCESS",
            data={"narration": narration, "events": [f"[{now_iso()}] generated by LLMBridge"], "error": None},
            metadata={"model": bridge.model, "base_url": bridge.base_url}
        )
    except Exception as e:
        return NarrateResponse(
            code="ERROR",
            data={"narration": "", "events": [], "error": str(e)},
            metadata={}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)

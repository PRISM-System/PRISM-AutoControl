# main.py — PRISM AutoControl Agent (API-only, with train/surrogate options)
import os, json, uuid, pathlib, logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from dotenv import load_dotenv

# ===== 외부 모듈/스키마 =====
from llm_io import LLMBridge
from autocontrol.schema import (
    AutoControlTargetSpec,
    AutoControlSpec,
    AutoControlRunRequest,
    AutoControlRunResponse,
    AutoControlRunResult,
    ControlCandidate,
    AC_NLRequest,
    AC_NLParsedResponse,
    NarrateRequest as AC_NarrateRequest,
    NarrateResponse as AC_NarrateResponse,
    NarrateData,
)

# 공용 유틸
from utils import (
    now_iso, rid, ensure_dirs,
    load_csv_and_features, ensure_target_in_features,
    ingest_weights, save_result_csv, artifact_download_url,
)

# 최적화/서러게이트
from optimizer import optimize_control
from surrogate import load_surrogate, train_and_save_linear_simple

from module import risk_module, explain_module
# ──────────────────────────────────────────────────────────────────────
# 환경/로깅/경로
# ──────────────────────────────────────────────────────────────────────
load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://147.47.39.144:8001/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "Qwen/Qwen3-14B")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "EMPTY")
API_PORT     = int(os.getenv("PORT", "8014"))

BASE_DIR           = pathlib.Path(__file__).resolve().parent
DATA_DIR_DEFAULT   = (BASE_DIR / "autocontrol" / "data" / "Industrial_DB_sample").resolve()
DEFAULT_CSV        = (DATA_DIR_DEFAULT / "SEMI_CMP_SENSORS.csv").resolve()
OUTPUTS_DIR        = (BASE_DIR / "outputs").resolve()
AUTOCONTROL_DIR    = (OUTPUTS_DIR / "autocontrol").resolve()
WEIGHTS_CACHE_DIR  = (OUTPUTS_DIR / "weights").resolve()
ensure_dirs([OUTPUTS_DIR, AUTOCONTROL_DIR, WEIGHTS_CACHE_DIR])

llm = LLMBridge(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY)

logger = logging.getLogger("autocontrol_agent")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(_handler)

def _notice(events: List[str], msg: str):
    logger.info(msg)
    events.append(f"[{now_iso()}] {msg}")

# ─────────────────────────────────────────────────────────────────────-
# FastAPI
# ─────────────────────────────────────────────────────────────────────-
app = FastAPI(title="PRISM AutoControl Agent (simple linear train)", version="1.4.0")

@app.get("/")
def root():
    return RedirectResponse(url="/healthz")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": now_iso()}

@app.get("/readyz")
def readyz():
    return {"ready": True, "llm_base_url": LLM_BASE_URL, "time": now_iso()}
@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204, content="")
# ─────────────────────────────────────────────────────────────────────-
# NL → JSON (원한다면 llm_io._extract_json_from_text로 대체 가능)
# ─────────────────────────────────────────────────────────────────────-
@app.post("/api/v1/autocontrol/nl/parse", response_model=AC_NLParsedResponse)
def ac_nl_parse(body: AC_NLRequest):
    events: List[str] = []
    _notice(events, f"[AC NL] query={body.query}")
    try:
        # 네 LLM 브리지가 오토컨트롤 전용 추출기가 없다면 _extract_json_from_text로 교체해도 됨
        raw = llm._extract_autocontrol_spec(body.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NL→JSON 실패: {e}")

    # CSV에서 feature 목록 확보(데모)
    csv_file, df, feature_df, csv_features = load_csv_and_features(None, DEFAULT_CSV, numeric_only=True)

    task_id   = raw.get("taskId") or f"ac_task_{uuid.uuid4().hex[:6]}"
    ac_id     = raw.get("acID") or f"ac_{uuid.uuid4().hex[:6]}"
    target_col = raw.get("target_col") or "MOTOR_CURRENT"
    if target_col not in csv_features:
        target_col = "MOTOR_CURRENT" if "MOTOR_CURRENT" in csv_features else csv_features[0]

    req_features = raw.get("feature_names")
    if isinstance(req_features, list) and req_features:
        feature_names = [f for f in req_features if f in csv_features] or [f for f in csv_features if f != target_col] or csv_features
    else:
        feature_names = [f for f in csv_features if f != target_col] or csv_features

    control = AutoControlTargetSpec(
        setpoint=float(raw.get("setpoint", 0.0)),
        horizon =int(raw.get("horizon", 11))
    )

    spec = AutoControlSpec(
        taskId=task_id,
        acID=ac_id,
        feature_names=feature_names,
        target_col=target_col,
        control=control,
        constraints=raw.get("constraints")
    )
    return {"code": "SUCCESS", "data": spec, "metadata": {"timestamp": now_iso(), "request_id": rid()}}

# ─────────────────────────────────────────────────────────────────────-
# Run Direct — 간단 학습/로딩/최적화
# ─────────────────────────────────────────────────────────────────────-
@app.post("/api/v1/autocontrol/run-direct", response_model=AutoControlRunResponse)
def ac_run_direct(
    body: AutoControlRunRequest,
    # 데이터/모델 입력
    csv_path: Optional[str] = Query(None, description="(옵션) CSV 경로 (미지정 시 기본 데모 CSV 사용)"),
    weight_path: Optional[str]  = Query(None, description="(옵션) 로컬 weight 경로(.npz)"),
    weight_url: Optional[str]   = Query(None, description="(옵션) weight 파일 URL(.npz)"),
    weights_b64: Optional[str]  = Query(None, description="(옵션) base64 인코딩된 weight(.npz)"),
    # 메타/선택
    model_meta: Optional[str]   = Query(None, description="(옵션) 응답에 표기할 모델명 메타"),
    surrogate: str              = Query("auto", description="서러게이트: auto | linear | fallback"),
    train: bool                 = Query(False, description="가중치가 있어도 재학습 강제"),
    train_if_missing: bool      = Query(True, description="가중치 없으면 새로 학습"),
):
    events: List[str] = []
    _notice(events, f"[AC] run-direct: taskId={body.taskId}, acID={body.acID}, target={body.target_col}, setpoint={body.control.setpoint}, horizon={body.control.horizon}")

    # 1) CSV 로드
    try:
        csv_file, df, feature_df, csv_features = load_csv_and_features(csv_path, DEFAULT_CSV, numeric_only=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"[AC] CSV 로드 실패: {e}")
    _notice(events, f"[AC] data loaded: rows={int(df.shape[0])}, cols={int(df.shape[1])}")

    # 2) 타깃/피처 검증
    try:
        ensure_target_in_features(feature_df, body.target_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"[AC] 타깃 컬럼 검증 실패: {e}")
    used_features = [f for f in body.feature_names if f in csv_features] or [f for f in csv_features if f != body.target_col]
    _notice(events, f"[AC] features used: {used_features[:8]}{'...' if len(used_features)>8 else ''}")

    # 3) 최근 상태 요약
    series  = pd.to_numeric(feature_df[body.target_col], errors="coerce").dropna().to_numpy()
    last_y  = float(series[-1]) if series.size >= 1 else 0.0
    delta_y = float(series[-1] - series[-2]) if series.size >= 2 else 0.0

    # 4) 가중치 인입(예측 에이전트 호환)
    weight_meta = ingest_weights(
        weight_path=weight_path,
        weight_url=weight_url,
        weights_b64=weights_b64,
        pred_activation_base_url=None,  # 단순화: 필요 시 다시 연결
        pred_task_id=None,
        weights_cache_dir=WEIGHTS_CACHE_DIR,
        events=events,
    )
    weight_local_path    = weight_meta.get("weight_local_path")
    best_model_from_pred = weight_meta.get("best_model")

    # 5) 서러게이트 결정: load vs train vs fallback (단순화)
    chosen_model_name = model_meta or best_model_from_pred
    expected_y_fn = None

    def _train_linear_and_load():
        nonlocal weight_local_path, chosen_model_name, expected_y_fn
        x_cols = [c for c in used_features if c != body.target_col]
        stub   = f"{body.taskId}_{body.acID}_linear_simple"
        ckpt   = train_and_save_linear_simple(
            feature_df, body.target_col,
            save_dir=WEIGHTS_CACHE_DIR,
            x_cols=x_cols,
            filename_stub=stub
        )
        weight_local_path = str(ckpt)
        chosen_model_name = chosen_model_name or "linear_simple_v1"
        expected_y_fn = load_surrogate(weight_local_path)
        _notice(events, f"[AC] linear(simple) trained & loaded: {pathlib.Path(ckpt).name}")

    if surrogate == "fallback":
        expected_y_fn = load_surrogate(None)
        chosen_model_name = chosen_model_name or "fallback"
        _notice(events, "[AC] using fallback surrogate")
    elif surrogate == "linear":
        if train or not weight_local_path:
            _train_linear_and_load()
        else:
            expected_y_fn = load_surrogate(weight_local_path)
            chosen_model_name = chosen_model_name or "linear_simple_v1"
            _notice(events, f"[AC] linear(simple) loaded from weight: {pathlib.Path(weight_local_path).name}")
    else:  # auto
        if weight_local_path and not train:
            expected_y_fn = load_surrogate(weight_local_path)
            chosen_model_name = chosen_model_name or best_model_from_pred or "linear_simple_v1"
            _notice(events, f"[AC] surrogate loaded from weight: {pathlib.Path(weight_local_path).name}")
        elif train_if_missing or train:
            _train_linear_and_load()
        else:
            expected_y_fn = load_surrogate(None)
            chosen_model_name = chosen_model_name or "fallback"
            _notice(events, "[AC] no weights; using fallback surrogate")

    # 6) 최적화 실행 (Top-K 후보)
    actuator_candidates = [f for f in used_features if f != body.target_col]
    cand_list = optimize_control(
        variables=actuator_candidates,
        last_y=last_y, delta_y=delta_y,
        setpoint=body.control.setpoint, horizon=body.control.horizon,
        constraints=body.constraints if isinstance(body.constraints, dict) else None,
        top_k=1,
        expected_y_fn=expected_y_fn,
        feature_df=feature_df, target_col=body.target_col,
        restarts=12, iters_per_restart=80,
    )

    cand_dicts = []
    for c in cand_list:
        c["setpoint_hint"] = float(body.control.setpoint)
        cand_dicts.append(c)

    # 7) 위험/설명
    risk = risk_module(feature_df, body.target_col, cand_dicts)
    if not isinstance(risk, dict):
        risk = {"byCandidate": risk}
    explanation = explain_module(feature_df, body.target_col, cand_dicts)

    # 8) 결과 CSV 저장 + 다운로드 URL
    out_rows = []
    for c in cand_list:
        row = {"candidate_id": c["id"], "expected_y": c["expected_y"], "score": c["score"]}
        for k, v in c["adjustments"].items():
            row[f"adj_{k}"] = v
        out_rows.append(row)
    try:
        result_csv_path   = save_result_csv(out_rows, AUTOCONTROL_DIR, body.taskId, body.acID)
        result_filename   = result_csv_path.name
        result_csv_public = artifact_download_url(result_filename)
        _notice(events, f"[AC] result CSV saved: {result_filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[AC] 결과 CSV 저장 실패: {e}")

    # 9) 응답
    data = AutoControlRunResult(
        taskId=body.taskId,
        acID=body.acID,
        input_csv_path=str(csv_file),
        result_csv_path=result_csv_public,
        feature_names=used_features,
        target_col=body.target_col,
        modelSelected=chosen_model_name,
        weight_path=weight_local_path or weight_meta.get("weight_remote_hint"),
        control=AutoControlTargetSpec(setpoint=body.control.setpoint, horizon=body.control.horizon),
        candidates=[ControlCandidate(**{k: v for k, v in c.items() if k in {"id","adjustments","expected_y","score"}}) for c in cand_list],
        selected_candidate_idx=0,
        risk=risk,
        explanation=explanation,
        events=events
    )
    return AutoControlRunResponse(
        code="SUCCESS",
        data=data,
        metadata={"timestamp": now_iso(), "request_id": rid()}
    )

# ─────────────────────────────────────────────────────────────────────-
# Narrate / Artifact
# ─────────────────────────────────────────────────────────────────────-
@app.post("/api/v1/autocontrol/narrate", response_model=AC_NarrateResponse)
def ac_narrate(body: AC_NarrateRequest):
    try:
        payload = dict(body.payload)
        if isinstance(payload.get("data"), dict):
            d = payload["data"]
            if isinstance(d.get("candidates"), list) and len(d["candidates"]) > 50:
                d["candidates"] = d["candidates"][:10] + [{"note": f"... {len(d['candidates'])-10} more candidates truncated"}]
        text = llm._narrate(payload)
        return AC_NarrateResponse(
            code="SUCCESS",
            data=NarrateData(narration=text, events=[f"[{now_iso()}] AC Narration OK"]),
            metadata={"timestamp": now_iso(), "request_id": rid()}
        )
    except Exception as e:
        return AC_NarrateResponse(
            code="ERROR",
            data=NarrateData(
                narration=llm._fallback_ko(body.payload),
                events=[f"[{now_iso()}] AC Narration failed: {e}"],
                error=str(e)
            ),
            metadata={"timestamp": now_iso(), "request_id": rid()}
        )

@app.get("/api/v1/autocontrol/artifacts/result")
def download_result(file: str = Query(..., description="결과 CSV 파일명")):
    target = (AUTOCONTROL_DIR / pathlib.Path(file).name).resolve()
    base = AUTOCONTROL_DIR
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=str(target), filename=target.name, media_type="text/csv")

# ─────────────────────────────────────────────────────────────────────-
# 로컬 실행
# ─────────────────────────────────────────────────────────────────────-
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)
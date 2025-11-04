import os, logging, requests
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv

from autocontrol.schema import (
    AutoControlRunRequest, AutoControlRunResponse, AutoControlRunResult,
)
from orchestra.schema import (
    OrchestrationAssignRequest, UpdatedAssignment, OrchestrationAssignResponse
)

from run import run_autocontrol

load_dotenv()
app = FastAPI(
    title="AutoControl API",
    description="시나리오 기반 Autocontrol 실행 api",
    version="1.1.0"
)

logger = logging.getLogger("prism_autocontrol")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

@app.on_event("startup")
def register_agent_on_startup():
    """PRISM-Core에 autonomous_control_agent를 등록합니다."""
    try:
        prism_core_url = os.getenv("PRISM_CORE_BASE_URL")
        if not prism_core_url:
            logger.error("PRISM_CORE_BASE_URL environment variable is not set")
            return
        # URL 끝의 슬래시 제거
        prism_core_url = prism_core_url.rstrip('/')
        logger.info(f"Registering autonomous_control_agent to PRISM-Core at {prism_core_url}")

        agent_data = {
            "name": "autonomous_control_agent",
            "description": "제조 공정의 자율 제어를 담당하는 자율제어 에이전트",
            "role_prompt": (
                "당신은 자율제어 에이전트입니다. 대리 모델 근사화(Surrogate Model Approximation)와 "
                "모델 예측 제어(MPC) 기법을 활용하여 제조 공정의 최적 제어 파라미터를 계산합니다. "
                "자율 의사결정 엔진으로 실시간 제어 전략을 수립하고, 위험 관리 및 설명 가능한 AI(XAI)를 통해 "
                "제어 결정의 근거를 명확히 제시합니다. 목표 성능은 0.220 RMSE 이하의 모델 근사 정확도와 "
                "99% 이상의 제어 성공률입니다. 제조 공정의 품질과 효율성을 최적화하고 안전성을 보장하세요."
            ),
            "tools": []
        }

        response = requests.post(
            f"{prism_core_url}/core/api/agents",
            json=agent_data,
            timeout=5
        )

        if response.status_code == 200:
            logger.info("✅ autonomous_control_agent registered successfully to PRISM-Core")
        else:
            logger.warning(f"⚠️ autonomous_control_agent registration failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to register autonomous_control_agent: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/docs")


@app.post("/api/v1/autocontrol/run-direct", response_model=AutoControlRunResponse)
def run(body: AutoControlRunRequest = Body(...)):
    # 경로는 env 또는 body.extra 등에서 받도록(유연)
    scenario_path = getattr(body, "scenario_path", None) or os.getenv("DEFAULT_SCENARIO_PATH", "scenarios/automotive/SCENARIO_11.json")
    data_path = getattr(body, "data_path", None) or os.getenv("DEFAULT_DATA_PATH", "./test_data/automotive/automotive_press_003.csv")

    out = run_autocontrol(
        scenario_path=scenario_path,
        data_path=data_path,
        feature_names=body.feature_names,
        target_col=body.target_col,
        control_setpoint=(body.control or {}).get("setpoint") if body.control else None,
        control_horizon_minutes=(body.control or {}).get("horizon") if body.control else None,
        constraints=body.constraints,
    )

    result = AutoControlRunResult(
        result_csv_path=out["result_csv_path"],
        candidates=out["candidates"],
        selected_candidate_idx=out["selected_candidate_idx"],
        nl_answer=out["nl_answer"],
    )
    return AutoControlRunResponse(
        task_id=body.taskId,
        ac_id=body.acID,
        data=result
    )


@app.put("/api/v1/task/{task_id}/autocontrol/assign", response_model=OrchestrationAssignResponse)
def orchestration_assign(req: OrchestrationAssignRequest = Body(...)):
    out = run_autocontrol(
        # scenario_path=os.getenv("DEFAULT_SCENARIO_PATH", "scenarios/automotive/SCENARIO_11.json"),
        data_path=os.getenv("DEFAULT_DATA_PATH", "./test_data/semiconductor/semiconductor_full_004.csv"),
        feature_names=req.timeseries_info["source_variables"],
        target_col=req.timeseries_info["target_variable"],
        control_setpoint=req.control_setpoint,
        control_horizon_minutes=req.control_horizon_minutes,
        constraints=req.constraints,
        query=req.query
    )

    payload = OrchestrationAssignResponse(
        task_id=req.taskId,
        updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="ready")],
        response={
            "autocontrol": {
                "summary": out["summary"],
                "result": out["nl_answer"],                         # <- main의 response를 그대로
                "controlled_timeseries": {
                    "format": "csv",
                    "description": f"제어 적용 후 제어 변수({out['extracted_X']})의 제어값과 목표 변수({out['extracted_y']})의 예측 값",
                    "sample_data": out["sample_data"]
                }
            }
        }
    )
    return payload





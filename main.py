import os, logging
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
def orchestration_assign(taskId: str, req: OrchestrationAssignRequest = Body(...)):
    # spec 구성 (요청에서 필요한 값만)
    control = getattr(req, "control", None) or {}
    out = run_autocontrol(
        # scenario_path=os.getenv("DEFAULT_SCENARIO_PATH", "scenarios/automotive/SCENARIO_11.json"),
        # data_path=os.getenv("DEFAULT_DATA_PATH", "./test_data/automotive/automotive_press_003.csv"),
        feature_names=getattr(req, "feature_names", None),
        target_col=getattr(req, "target_col", None),
        control_setpoint=control.get("setpoint"),
        control_horizon_minutes=control.get("horizon"),
        constraints=getattr(req, "constraints", None),
    )

    payload = OrchestrationAssignResponse(
        task_id=req.task_id or taskId,
        updated_assignments=[UpdatedAssignment(agent_id="autocontrol", status="ready")],
        response={
            "autocontrol": {
                "nl_answer": out["nl_answer"],                         # <- main의 response를 그대로
                "result_csv_path": out["result_csv_path"],
                "selected_candidate": {
                    "id": "cand_0",
                    "adjustments": {X: out["candidates"][X][0] for X in out["extracted_X"]} 
                },
                "expected_y": out["candidates"][out["extracted_y"]][0],
                "score": 0.9
            }
        }
    )
    return payload





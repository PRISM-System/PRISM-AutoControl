import os, json, logging
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import casadi as ca
import do_mpc
import matplotlib.pyplot as plt

from llm_io import LLMBridge
from utils import *
from optimization.utils import *
from optimization.do_mpc import *

def run_autocontrol(
    data_path: str,
    feature_names: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    control_setpoint: Optional[float] = None,
    control_horizon_minutes: Optional[float] = None,
    constraints: Optional[Dict[str, Dict[str, float]]] = None,
    optimization_objective: Optional[str] = None,
) -> Dict[str, Any]:
    """
    코어 실행 함수: ARX 적합 + do-mpc 세팅/최적화 + LLM 설명 생성
    반환: {'nl_answer': str, 'result_csv_path': str, 'candidates': [], 'selected_candidate_idx': None, ...}
    """
    # --- 시나리오/데이터 로드 ---
    # scenario = load_scenario(scenario_path)
    input_data = pd.read_csv(data_path)

    # 샘플링 주기(초)
    dt = infer_dt_seconds(input_data)
    
    X = feature_names
    y = target_col
    target_setpoint = control_setpoint
    horizon_min = control_horizon_minutes
    X_constraints = constraints
    
    # --- 컬럼 매핑 ---
    dataset_columns = list(input_data.drop('TIMESTAMP', axis=1).columns)
    extracted_X, extracted_y, mapping_score = extract_features_from_query(X, y, dataset_columns)
    extracted_X = [mv for mv in extracted_X if mv is not None]
    if extracted_y is None:
        raise ValueError("Target column could not be mapped from query.")

    # --- Horizon 계산 ---
    n_rows = len(input_data)
    total_minutes = (max(n_rows - 1, 1) * dt) / 60.0
    prediction_horizon_minutes = float(horizon_min) * 2.0
    history_horizon_minutes = min(float(horizon_min) * 4.0, total_minutes)

    pred_steps = int(round(prediction_horizon_minutes * 60.0 / dt))
    ctrl_steps = int(round(float(horizon_min) * 60.0 / dt))
    window_samples = int(round(history_horizon_minutes * 60.0 / dt))

    # --- ARX 적합 ---
    a, b, c, last = fit_arx(input_data, extracted_X, extracted_y, window=window_samples)

    # --- do-mpc 모델/컨트롤러 ---
    model = do_mpc.model.Model('discrete')
    x = model.set_variable('_x', 'x')
    u_vars = [model.set_variable('_u', mv) for mv in extracted_X]
    x_next = a*x + sum(b[i]*u_vars[i] for i in range(len(extracted_X))) + c
    model.set_rhs('x', x_next)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(n_horizon=pred_steps, t_step=dt, store_full_solution=True)

    sp = float(target_setpoint) if target_setpoint is not None else float(input_data[extracted_y].mean())
    mpc.set_objective(mterm=(x-sp)**2, lterm=(x-sp)**2)
    Ru = 1e-3
    for mv in extracted_X:
        mpc.set_rterm(**{mv: Ru})

    x0 = float(last[extracted_y])
    u0 = np.array([float(last[mv]) for mv in extracted_X], dtype=float)
    mpc.x0 = x0
    mpc.u0 = u0
    mpc.setup()

    # --- 제약(데이터 범위) ---
    u_bounds = {}
    for mv in extracted_X:
        lo = float(input_data[mv].min())
        hi = float(input_data[mv].max())
        u_bounds[mv] = (lo, hi)
        mpc.bounds['lower','_u', mv] = lo
        mpc.bounds['upper','_u', mv] = hi

    # (선택) 실제 constraints 적용
    if X_constraints:
        for key, cons in X_constraints.items():
            info = mapping_score.get(key.lower(), {})
            mapped_col = info.get('matched_column')
            if not mapped_col or mapped_col not in extracted_X:
                continue
            if 'min' in cons:
                mpc.bounds['lower','_u', mapped_col] = float(cons['min'])
                u_bounds[mapped_col] = (float(cons['min']), u_bounds[mapped_col][1])
            if 'max' in cons:
                mpc.bounds['upper','_u', mapped_col] = float(cons['max'])
                u_bounds[mapped_col] = (u_bounds[mapped_col][0], float(cons['max']))

    # --- 예측/플랜 ---
    x_pred, u_pred = MPC_predictions(mpc, x0, extracted_X)
    t_idx = np.arange(pred_steps) * dt

    df_future = pd.DataFrame({'t_sec': t_idx, 't_min': t_idx/60.0})
    for j, mv in enumerate(extracted_X):
        df_future[mv] = u_pred[:, j]
    df_future[extracted_y] = x_pred[:pred_steps]

    hist_len = int(min(window_samples, len(input_data) - 1))
    hist_slice = input_data.iloc[-hist_len-1:-1].copy()
    t_hist_idx = -np.arange(hist_len, 0, -1)
    t_hist_sec = t_hist_idx * dt
    df_hist = pd.DataFrame({'t_sec': t_hist_sec, 't_min': t_hist_sec/60.0})
    df_hist[f'{extracted_y}_hist'] = hist_slice[extracted_y].astype(float).to_numpy()
    for mv in extracted_X:
        df_hist[mv + '_hist'] = hist_slice[mv].astype(float).to_numpy()

    # --- CSV 저장(선택) ---
    result_csv_path = "./results/default/mpc_plan.csv"
    Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_future.to_csv(result_csv_path, index=False)

    # --- LLM 설명 생성 ---
    llm = LLMBridge(username="kaist", password="kaist1234", verify=False)
    llm.login()
    prompt = (
        f"사용자가 입력한 제어 변수명은 {X}, 타겟 변수명은 {y}였어. "
        f"하지만 실제로 데이터에 그 변수는 없었고, 코사인 유사도 기반으론 제어 변수 명은 {extracted_X}, "
        f"타겟 변수 명은 {extracted_y}로 예상돼. "
        f"이 변수들을 사용하여 향후의 {pred_steps}을 예측하여 {ctrl_steps} step만큼을 제어하고자 Model predictive control 모델을 돌렸고, "
        f"제어 세팅의 초기 상태는 {x0}, 가장 초기의 입력은 {u0}야. 최적화하고자 하는 타겟 setpoint는 {sp}이고, "
        f"이 때 제어 변수의 constraints는 다음과 같아 : {u_bounds}. "
        f"참고한 과거의 데이터 일부: {df_hist.head(5).to_dict()}, "
        f"최적화 결과 후보 일부: {df_future.head(5).to_dict()}. "
        f"후보들의 장점, 단점, 위험성, 기대 효과를 정리해줘."
    )
    nl_answer = llm.narrate(prompt)

    # 호출측(FastAPI)이 그대로 넣어 쓸 수 있게 dict로 반환
    return {
        "nl_answer": nl_answer,
        "result_csv_path": result_csv_path,
        "candidates": df_future.head(1).to_dict(),                # 필요 시 채워 넣기
        "selected_candidate_idx": None,  # 필요 시 채워 넣기
        "extracted_X": extracted_X,
        "extracted_y": extracted_y,
        "setpoint": sp,
        "x0": x0,
        "u0": u0.tolist(),
        "u_bounds": u_bounds,
        "pred_steps": pred_steps,
        "ctrl_steps": ctrl_steps,
    }


if __name__ == "__main__":
    # 로컬 실행(개발용): 기존 코드와 동일 동작
    scenario_path = "scenarios/automotive/SCENARIO_11.json"
    data_path = "./test_data/automotive/automotive_press_003.csv"

    out = run_autocontrol(
        scenario_path=scenario_path,
        data_path=data_path,
        feature_names=None,            # 시나리오에서 읽음
        target_col=None,               # 시나리오에서 읽음
        control_setpoint=None,         # 시나리오에서 읽음
        control_horizon_minutes=None,  # 시나리오에서 읽음
        constraints=None,
    )
    print("\n=== LLM 응답 ===")
    print(out["nl_answer"])


#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://127.0.0.1:8100/api/v1"

TASK_ID = "cmp_task_001"

def call_orchestration_assign():
    url = f"{BASE_URL}/task/{TASK_ID}/autocontrol/assign"
    payload = {
        "task_id": TASK_ID,
        "agent_assignments": [
            {
                "agent_id": "autocontrol",
                "agent_type": "planner",
                "execution_order": {
                    "nl_query": (
                        "CMP 센서의 MOTOR_CURRENT가 너무 높은 상황이야. "
                        "HEAD_ROTATION과 SLURRY_FLOW_RATE을 조정하여 "
                        "MOTOR_CURRENT를 1구간 내에 17.5까지 줄여야 해."
                    ),
                    "surrogate_model": "sklearn:random_forest_regressor",
                    "surrogate_params": "{\"n_estimators\":200}",
                    "opt_method": "lbfgsb",
                    "maxiter": 20,
                    "agent_base_url": None,
                    "agent_task_id": "autocontrol_1",
                    "agent_inline": True,
                    "weight_local_path": "string"
                }
            }
        ]
    }
    r = requests.put(url, json=payload)
    print("[assign] status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text)
    r.raise_for_status()


def call_run_direct():
    url = f"{BASE_URL}/autocontrol/run-direct"
    payload = {
        "taskId": TASK_ID,
        "acID": "ac_7b32d9",
        "feature_names": ["HEAD_ROTATION", "SLURRY_FLOW_RATE"],
        "target_col": "MOTOR_CURRENT",
        "control": {"setpoint": 17.5, "horizon": 10},
        "constraints": {
            "bounds": {
                "HEAD_ROTATION": {"min": -50, "max": 50, "step": 1.0},
                "SLURRY_FLOW_RATE": {"min": -200, "max": 200, "step": 5.0},
            }
        },
        "csv_path": "/mnt/c/Users/chanh/Desktop/IITP_2025/PRISM-AutoControl/autocontrol/data/Industrial_DB_sample/SEMI_CMP_SENSORS.csv",
        "fromAgent": "orch",
        "objective": "control",
    }
    params = {
        "surrogate_model": "sklearn:random_forest_regressor",
        "surrogate_params": "{\"n_estimators\":200}",
        "opt_method": "lbfgsb",
        "maxiter": 20,
    }
    r = requests.post(url, json=payload, params=params)
    print("[run-direct] status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text)
    r.raise_for_status()


def main():
    print("[assign+run] 시작 …")
    call_orchestration_assign()
    call_run_direct()
    print("[done]")


if __name__ == "__main__":
    main()

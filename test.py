import requests, json, os

BASE = "http://127.0.0.1:8100" 

nl = (
    "CMP 센서의 MOTOR_CURRENT가 너무 높은 상황이야. "
    "SLURRY_FLOW_RATE을 조정하여 MOTOR_CURRENT를 14구간 내에 12까지 줄여야 해. "
)

r1 = requests.post(f"{BASE}/api/v1/autocontrol/parse-nl", json={"query": nl})
r1.raise_for_status()
parsed = r1.json()
spec = parsed["data"]  
print("[parse-nl] OK")
print(json.dumps(parsed, ensure_ascii=False, indent=2))

params = {
    "surrogate_model": "sklearn:random_forest_regressor",    # 또는 "sklearn:random_forest_regressor"
    "opt_method": "lbfgsb",     # 또는 "powell", "de", "slsqp", ...
    # --- 예측 에이전트 스냅샷을 써서 가중치 받으려면 아래 주석 해제 ---
    # "agent_base_url": "http://127.0.0.1:8001",
    # "agent_task_id": spec.get("taskId"),  # parse 결과의 taskId 사용
    # "agent_inline": "true",
}

r2 = requests.post(f"{BASE}/api/v1/autocontrol/run-direct", params=params, json=spec)
r2.raise_for_status()
out = r2.json()
print("\n[run-direct] OK")

print(json.dumps(
    {
        "modelSelected": out["data"]["modelSelected"],
        "candidates": [out["data"]["candidates"][0]] if out["data"]["candidates"] else [],
        "result_csv_path": out["data"]["result_csv_path"],
    },
    ensure_ascii=False, indent=2
))
print("\n[nl-answer]", out.get("metadata", {}).get("nl_answer"))
result_path = out["data"]["result_csv_path"]

def save_bytes_to(fname: str, content: bytes):
    with open(fname, "wb") as f:
        f.write(content)
    print(f"[result.csv] saved -> {fname}")

if isinstance(result_path, str) and result_path.startswith("/api/"):
    resp = requests.get(f"{BASE}{result_path}")
    
    resp.raise_for_status()
    save_bytes_to("result.csv", resp.content)
else:
    if not os.path.exists(result_path):
        raise SystemExit(f"result_csv_path not found: {result_path}")
    with open(result_path, "rb") as f:
        save_bytes_to("result.csv", f.read())

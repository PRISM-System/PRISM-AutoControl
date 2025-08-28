# test.py
import requests, json

# 서버가 같은 WSL 터미널이면 127.0.0.1 사용
# Windows PowerShell에서 테스트면 보통 localhost가 됨 (안 되면 WSL IP로 교체)
BASE = "http://127.0.0.1:8014"  # 필요시 http://localhost:8014 로 바꾸세요

# 1) NL → JSON
nl = "CMP 센서의 MOTOR_CURRENT가 너무 높은 상황이야. HEAD_ROTATION을 조정하여 MOTOR_CURRENT를 1구간 내에 13.0까지 줄여야 해."
r1 = requests.post(f"{BASE}/api/v1/autocontrol/nl/parse", json={"query": nl})
r1.raise_for_status()
spec = r1.json()["data"]
print("[nl/parse] OK")
print(json.dumps(spec, ensure_ascii=False, indent=2))

# 2) run-direct (가중치 없으면 자동으로 선형 학습)
params = {"surrogate": "auto", "train_if_missing": "true"}
# 기본 CSV가 서버에 없으면 아래처럼 절대경로 지정:
# params["csv_path"] = "/절대/경로/SEMI_CMP_SENSORS_predict.csv"

r2 = requests.post(f"{BASE}/api/v1/autocontrol/run-direct", params=params, json=spec)
r2.raise_for_status()
out = r2.json()
print("\n[run-direct] OK")
print(json.dumps({k: out["data"][k] for k in ["modelSelected","candidates","result_csv_path"]},
                 ensure_ascii=False, indent=2))

# 3) 결과 CSV 저장
rel = out["data"]["result_csv_path"]  # 예: /api/v1/autocontrol/artifacts/result?file=...
csv = requests.get(f"{BASE}{rel}")
csv.raise_for_status()
open("result.csv","wb").write(csv.content)
print("\n[result.csv] saved")

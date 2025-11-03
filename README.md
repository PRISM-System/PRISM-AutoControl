# PRISM-AutoControl: 자율제어 AI 에이전트

`PRISM-AutoControl`은 [PRISM-AGI](../README.md) 플랫폼의 자율제어를 담당하는 AI 에이전트입니다. 예측된 결과를 바탕으로 최적의 제어 액션을 스스로 결정하고 실행합니다.

---

## 1. 주요 기능

### 대체 모델 근사 시스템
- 복잡한 AI 모델의 동작을 근사하여 빠른 연산을 지원하는 모듈
- 시뮬레이션 환경에서 제어 액션의 유효성과 안정성을 사전에 검증
- 제한된 연산 자원에서도 효율적으로 동작하기 위한 모델 경량화
- 제어 액션이 물리 법칙을 준수하는지 검증하는 시스템

### 자율 의사결정 시스템
- 센서 데이터를 통해 실시간으로 공정 상태를 정확하게 인식하는 모듈
- 목표 달성을 위한 최적의 전략을 수립하는 자율제어 의사결정 엔진
- 현재 상태에서 실행 가능한 제어 액션 후보군을 생성하는 알고리즘
- 후보군 중에서 최적의 액션을 선택하고 실행하는 시스템

### 위험 관리 및 설명
- 자율적인 의사결정에 따르는 잠재적 위험을 정량적으로 평가
- 결정된 제어 액션의 이유와 근거를 인간이 이해할 수 있는 형태로 설명 (XAI)
- 제어 액션의 근거 데이터를 투명하게 제공하는 시스템
- 자율 최적화 과정의 성공률을 지속적으로 관리하고 개선

### 소프트웨어 개발 최적화
- 복잡한 제어 과업을 분해하고 역할을 분할하여 개발 효율성 증대
- 병렬 처리 및 최적화를 통해 시스템의 전반적인 성능 향상

---

## 2. 성능 목표

| 기능           | 지표                     | 목표       |
| ---           | ---                      | ---       |
| **제어 정확도** | AI 모델 근사 정확도 (RMSE) | 0.220 이하 |
|               | 자율제어 의사결정 예측 오차  | 10% 이내   |
| **신뢰성**     | 자율 최적화 성공률          | 99%       |
| **위험 관리**  | 의사결정 위험 평가 상관계수   | 0.5 이상  |
| **개발 효율**  | 작업 분해 및 역할 분할 효율성 | 10% 향상  |
|               | 병렬 처리 및 최적화 성능     | 10% 향상  |

---

## 3. 시연용 절차

### 3-1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3-2. 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 4-1. 서버를 실행한 이후, 시나리오 step6의 request에 해당하는 json을 입력하여 출력 확인

```bash
curl -X POST http://localhost:8001/api/v1/prediction/run-direct   -H "Content-Type: application/json"   --data-binary @- <<'JSON'
{
  "step_4_orchestration_to_prediction": {
    "from": "Orchestration",
    "to": "Predictive",
    "timestamp": "2025-05-01T14:20:07Z",
    "api_endpoint": "POST /api/v1/prediction/run-direct",
    "request": {
      "taskId": "ETCH_TASK_20250501_002_2",
      "timeRange": {
        "start": "2025-05-01T12:50:00Z",
        "end": "2025-05-01T14:20:00Z"
      },
      "sensor_name": "ETCH_CH1,ETCH_CH2,ETCH_CH3,ETCH_CH4",
      "target_cols": ["PRESSURE", "PROCESS_QUALITY_INDEX"],
      "feature_cols": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE"],
      "prediction_horizon_minutes": 90,
      "prediction_interval_minutes": 5,
      "confidence_level": 0.95
    }
  }
}
JSON
```

### 엔드포인트
```
POST /api/v1/prediction/run-direct
```

## 응답 예시 (실제 출력)

아래는 예측 API 호출 시의 예시 응답입니다.  
모델/버전/데이터에 따라 수치는 달라질 수 있습니다.

```json
{
  "code": "SUCCESS",
  "data": {
    "result": "# 산업 공정 예측 리포트\n\n## 1. 개요\n현재 압력이 8.7 mTorr로 정상 범위(5.0-7.0 mTorr)를 24.3% 초과하여 지속 상승 중입니다. 진공 펌프의 효율이 72.3%로 저하되어 압력 상승의 주요 원인으로 확인되었습니다. 현재 압력 상승률은 +0.028 mTorr/분입니다.\n\n## 2. 예측 결과\n향후 90분간의 압력 및 진공 펌프 효율 예측 결과는 다음과 같습니다.\n\n| 시간 (UTC) | 압력 (mTorr) | 진공 펌프 효율 (%) |\n|-------------|--------------|---------------------|\n| 2025-05-01 14:25 | 93.83 | 90.07 |\n| 2025-05-01 14:30 | 93.58 | 89.92 |\n| 2025-05-01 14:35 | 92.86 | 90.07 |\n| 2025-05-01 14:40 | 94.16 | 89.94 |\n| 2025-05-01 14:45 | 93.57 | 90.02 |\n| 2025-05-01 14:50 | 92.54 | 90.03 |\n| 2025-05-01 14:55 | 92.48 | 90.06 |\n| 2025-05-01 15:00 | 92.94 | 90.04 |\n| 2025-05-01 15:05 | 92.26 | 89.95 |\n| 2025-05-01 15:10 | 93.24 | 90.11 |\n| 2025-05-01 15:15 | 93.72 | 89.99 |\n| 2025-05-01 15:20 | 93.45 | 90.05 |\n| 2025-05-01 15:25 | 92.64 | 90.04 |\n| 2025-05-01 15:30 | 93.90 | 89.93 |\n| 2025-05-01 15:35 | 92.63 | 90.09 |\n| 2025-05-01 15:40 | 93.12 | 90.04 |\n| 2025-05-01 15:45 | 93.21 | 89.95 |\n| 2025-05-01 15:50 | 93.18 | 89.97 |\n\n## 3. 임계치 도달 분석\n- **임계치(10.0 mTorr) 도달 시점**: 예측된 압력은 90분 후에도 10.0 mTorr에 도달하지 않을 것으로 보입니다. 그러나 현재 압력이 이미 정상 범위를 초과하고 있어, 지속적인 모니터링이 필요합니다.\n- **인터록 작동 가능성**: 현재 위험 수준은 \"높음\"으로 평가되며, 압력이 계속 상승할 경우 인터록 작동 가능성이 존재합니다. 따라서 즉각적인 조치가 필요합니다.\n\n## 4. 결론\n압력 상승이 지속되고 있으며, 진공 펌프의 효율 저하가 주요 원인으로 확인되었습니다. 향후 90분간의 예측 결과에 따르면, 압력이 10.0 mTorr에 도달하지는 않겠지만, 현재의 높은 위험 수준을 고려할 때 즉각적인 조치가 필요합니다.",
    "raw": {
      "spec": {
        "taskId": "ETCH_TASK_20250501_002_2",
        "query": "모니터링 결과 압력이 8.7 mTorr로 정상 범위(5.0-7.0 mTorr)를 24.3% 초과하여 지속 상승 중입니다. 진공 펌프 효율이 72.3%로 저하되어 압력 상승의 주요 원인으로 확인되었습니다. 현재 상승률(+0.028 mTorr/분)이 계속될 경우 향후 90분간 압력과 공정 품질이 어떻게 전개될지 예측해주세요. 특히 임계치(10.0 mTorr) 도달 시점과 인터록 작동 가능성을 분석해주세요.",
        "timeRange": {
          "start": "2025-05-01T12:50:00Z",
          "end": "2025-05-01T14:20:00Z"
        },
        "sensor_name": "CHAMBER_E1,CHAMBER_E2,CHAMBER_E3,CHAMBER_E4",
        "target_cols": ["PRESSURE", "VACUUM_PUMP"],
        "feature_cols": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE"],
        "prediction_horizon_minutes": 90,
        "prediction_interval_minutes": 5,
        "model_type": "lstm",
        "confidence_level": 0.95
      },
      "csv_path": "prism_prediction/Industrial_DB_sample/dataset_v3/test_scenarios/test_data/semiconductor/semiconductor_etch_002.csv",
      "df_info": { "rows": 5000, "cols": 11 },
      "feature_names": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE","ETCH_RATE","BIAS_VOLTAGE","CHAMBER_HUMIDITY","GAS_COMPOSITION"],
      "enc_in": 9,
      "target_col": "PRESSURE",
      "target_idx_in_features": 0,
      "pred_len": 18,
      "confidence_level": 0.95,
      "sensor_name": "CHAMBER_E1,CHAMBER_E2,CHAMBER_E3,CHAMBER_E4",
      "risk": { "riskLevel": "high", "exceedsThreshold": true },
      "explanation": {
        "importantFeatures": ["GAS_FLOW_RATE","ETCH_RATE","TEMPERATURE","RF_POWER","BIAS_VOLTAGE"],
        "method": "corr-proxy"
      }
    }
  },
  "metadata": {
    "timestamp": "2025-11-03T00:29:25Z",
    "request_id": "req_6be6d296"
  }
}

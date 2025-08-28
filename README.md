# PRISM-AutoControl: 자율제어 AI 에이전트

`PRISM-AutoControl`은 [PRISM-AGI](../README.md) 플랫폼의 자율제어를 담당하는 AI 에이전트입니다. 예측된 결과를 바탕으로 최적의 제어 액션을 스스로 결정하고 실행합니다.

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

## 2. 성능 목표

| 기능           | 지표                     | 목표       |
| ---           | ---                      | ---       |
| **제어 정확도** | AI 모델 근사 정확도 (RMSE) | 0.220 이하 |
|               | 자율제어 의사결정 예측 오차  | 10% 이내   |
| **신뢰성**     | 자율 최적화 성공률          | 99%       |
| **위험 관리**  | 의사결정 위험 평가 상관계수   | 0.5 이상  |
| **개발 효율**  | 작업 분해 및 역할 분할 효율성 | 10% 향상  |
|               | 병렬 처리 및 최적화 성능     | 10% 향상  |

## 3. 설치 및 실행 가이드

### 3.1 시스템 요구사항

- **Python**: 3.9 이상
- **OS**: Windows 10/11, Ubuntu 20.04 이상, macOS 11 이상

### 3.2 로컬 개발 환경 설정

#### 3.2.1 저장소 클론
```bash
git clone https://github.com/your-org/prism-autocontrol.git
cd prism-autocontrol
```

#### 3.2.2 Python 가상환경 설정
```bash
# Windows (cmd)
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

#### 3.2.3 의존성 설치
```bash
pip install -r requirements.txt
```

### 3.3 PRISM-AutoControl 서버 실행 (새 cmd 창에서 실행)
```bash
venv\Scripts\activate
python -m uvicorn main:app --reload --port 8100 --host 127.0.0.1
```

#### 3.3.1 헬스체크 (새 cmd 창에서 실행)
```bash
curl -v http://127.0.0.1:8100/healthz
```

#### 3.3.2 PRISM-AutoControl 에이전트에 질의 ***현재는 답변을 임의의 string 중 하나를 뽑는 것으로 구현; random_seed=42로 고정*** (헬스체크한 기존 cmd 창에서 입력)
```bash
curl -v -X POST "http://127.0.0.1:8100/api/v1/autocontrol/nl" ^
  -H "Content-Type: application/json" ^
  -d "{""prompt"":""플래튼 회전 속도와 슬러리 온도 제어 밸브는 어떻게 조정할까?"", ""seed"": 42}"
```

#### 3.3.3 PRISM-Orchestration 에이전트를 통해 PRISM-AutoControl에 질의 (새 cmd 창에서 실행)
```bash
venv\Scripts\activate
python -m uvicorn orch_stub:app --reload --host 127.0.0.1 --port 8801
```

```bash
curl -v -X POST "http://127.0.0.1:8801/orchestrate" ^
  -H "Content-Type: application/json" ^
  -d "{""prompt"":""최근 CMP 공정 온도가 이상 추세. 15분 내 정상화 권고 요청."", ""seed"": 42}"
```


### Appendix. 에이전트 모듈 통신

#### 1. PRISM-Core 에이전트 통신
```bash
curl -v -X POST "http://147.47.39.144:8000/api/generate" ^
  -H "Content-Type: application/json" ^
  -d "{""prompt"":""안녕하세요! 간단한 자기소개를 해주세요.""}"
```

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class AutoControlTargetSpec(BaseModel):
    setpoint: float = Field(..., description="목표 y 값 (제어 변수의 목표값)")
    horizon: int = Field(11, ge=1, description="제어 구간 (미래 몇 step을 제어할지)")

class AutoControlSpec(BaseModel):
    """
    AutoControlSpec: 하나의 제어 동작 요청을 정의하는 스키마
    - 하나의 사용자 질의(taskId) 안에 여러 제어 동작(acID)이 있을 수 있음
    """
    taskId: str = Field(..., description="질의 고유 식별자 (사용자 질문 단위)")
    acID: str = Field(..., description="제어 동작 식별자 (질의 내에서 개별 제어 단위를 구분)")
    feature_names: List[str] = Field(..., description="제어 대상이 되는 Manipulated Variable (조작 변수) 이름")
    target_col: str = Field(..., description="제어 타겟이 되는 Controlled Variable (제어 변수) 이름")
    control: AutoControlTargetSpec = Field(..., description="제어 목표 값(setpoint) 및 제어 구간(horizon) 정의")
    constraints: Optional[Any] = Field(None, description="제어 제약 조건")
    csv_path : Optional[str] = Field(None, description="Local DB CSV 경로 (다른 agent랑 연결했을 때도 필요할지는 모르겠음)")
class AutoControlRunRequest(AutoControlSpec):
    """
    AutoControlRunRequest: 제어 실행 요청 스키마
    """
    fromAgent: Literal["orch", "prediction", "monitoring", "external"] = Field(
        "orch", description="요청 발생 주체"
    )
    objective: Literal["control", "autocontrol", "제어"] = Field(
        "control", description="요청 목적"
    )

class ControlCandidate(BaseModel):
    """
    ControlCandidate: 제어 동작 후보군 스키마
    """
    id: str = Field(..., description="후보군 고유 식별자")
    adjustments: Dict[str, float] = Field(..., description="각 조작 변수별 조정값")
    expected_y: float = Field(..., description="해당 제어 적용 시 예상되는 y (제어 변수 값)")
    score: float = Field(..., description="제어 적합 점수 (목표와의 오차/효율성 기반)")

class AutoControlRunResult(BaseModel):
    """
    AutoControlRunResult: 제어 실행 결과 스키마
    """
    taskId: str = Field(..., description="질의 고유 식별자 (사용자 질문 단위)")
    acID: str = Field(..., description="제어 동작 식별자 (질의 내에서 개별 제어 단위를 구분)")
    input_csv_path: str = Field(..., description="제어 데이터 csv 경로")
    result_csv_path: str = Field(..., description="실행 로그/결과 CSV 경로")
    feature_names: List[str] = Field(..., description="제어 대상이 되는 Manipulated Variable (조작 변수) 이름")
    target_col: str = Field(..., description="제어 타겟이 되는 Controlled Variable (제어 변수) 이름")
    modelSelected: Optional[str] = Field(None, description="선택된 모델 이름 (있을 경우)")
    weight_path: Optional[str] = Field(None, description="모델 weight 경로 (있을 경우)")
    control: AutoControlTargetSpec = Field(..., description="제어 목표 값(setpoint) 및 제어 구간(horizon) 정의")
    candidates: List[ControlCandidate] = Field(..., description="생성된 제어 후보군 목록")
    selected_candidate_idx: int = Field(..., description="최종 선택된 후보 index")
    risk: Dict[str, Any] = Field(..., description="제어 위험 평가 결과")
    explanation: Dict[str, Any] = Field(..., description="제어 설명/해석 결과")
    events: List[str] = Field(..., description="제어 중 발생 이벤트 로그")

class AutoControlRunResponse(BaseModel):
    """
    AutoControlRunResponse: 제어 실행 API 응답 스키마
    """
    code: Literal["SUCCESS","ERROR"] = Field("SUCCESS", description="제어 실행 결과 상태 코드")
    data: AutoControlRunResult = Field(..., description="제어 실행 결과")
    metadata: Dict[str, str] = Field(..., description="부가 메타데이터")
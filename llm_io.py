import os, json, time, re, requests
from typing import Any, Dict, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any, Dict, List, Optional
import difflib, re, logging
import logging
logger = logging.getLogger(__name__)
def _is_bound_dict(v) -> bool:
    return isinstance(v, dict) and any(k in v for k in ("min", "max", "step"))
class LLMUnavailable(Exception):
    """LLM 서버 연결/타임아웃/5xx 등 가용성 문제"""
    pass


class LLMBadResponse(Exception):
    """LLM가 비JSON/이상 응답을 반복해서 주는 문제"""
    pass
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _tok(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]

def _context_tokens(nl_query: str):
    return set(_tok(nl_query or ""))

def _qualifier_words():
    # 도메인 보정 단어. 필요시 확장하세요.
    return {"head", "platen", "retainer", "upper", "lower", "wafer",
            "slurry", "chuck", "carrier", "pad", "table", "chamber"}

def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _best_feature_match(name: str, available_features, nl_query: str) -> str | None:
    """
    주어진 name을 available_features 중 최적 후보로 매핑.
    - 대소문자 무시 완전일치 우선
    - 유사도 + 토큰포함 + 접두/접미 보너스 + 컨텍스트(qualifier) 보너스
    - 2위와 0.05 이내면 모호하다고 로그 경고 후 1위 선택
    """
    logger = logging.getLogger(__name__)
    if not name:
        return None
    if not available_features:
        return name

    # 1) 대소문자 무시 완전 일치
    for f in available_features:
        if f.lower() == name.lower():
            return f

    ctx = _context_tokens(nl_query)
    quals = _qualifier_words()

    cand = []
    name_tokens = set(_tok(name))
    for f in available_features:
        base = _similarity(name, f)
        bonus = 0.0

        # 부분문자열/접두/접미
        nf = _norm(f)
        nn = _norm(name)
        if nn and nn in nf:
            bonus += 0.15
        if nf.endswith(nn) or nf.startswith(nn):
            bonus += 0.10

        # 토큰 교집합
        ftoks = set(_tok(f))
        overlap = len(name_tokens & ftoks)
        if overlap:
            bonus += min(0.10 * overlap, 0.30)

        # 컨텍스트 단서(qualifier)
        ctx_overlap = (ctx & quals & ftoks)
        if ctx_overlap:
            # 단서가 많을수록 소폭 보너스
            bonus += min(0.20 + 0.05 * (len(ctx_overlap) - 1), 0.30)

        cand.append((base + bonus, f))

    cand.sort(reverse=True, key=lambda x: x[0])
    if not cand:
        return name

    best = cand[0]
    second = cand[1] if len(cand) > 1 else (0.0, None)
    if second[1] and (best[0] - second[0]) < 0.05:
        logger.warning("[feature-match] ambiguous for '%s' -> %s; picked '%s'",
                       name, [c[1] for c in cand[:3]], best[1])
    return best[1]


def _reconcile_and_harden_features(spec: dict, nl_query: str, available_features: list[str] | None) -> dict:
    """
    - target_col, feature_names, constraints.bounds 키를 available_features로 매핑
    - bounds는 가용 컬럼 외의 키 제거
    - feature_names에 target_col/바운드 키 보강
    """
    logger = logging.getLogger(__name__)
    if not spec:
        return spec
    if not available_features:
        return spec

    af = list(available_features)
    aset = set(af)

    # target_col 매핑
    tc = spec.get("target_col")
    mtc = _best_feature_match(tc, af, nl_query) if tc else None
    if mtc and mtc != tc:
        logger.info("[feature-reconcile] target_col '%s' -> '%s'", tc, mtc)
        spec["target_col"] = mtc

    # feature_names 매핑
    fn = spec.get("feature_names") or []
    new_fn = []
    for n in fn:
        m = _best_feature_match(n, af, nl_query)
        if m and m not in new_fn:
            new_fn.append(m)

    # bounds 키 매핑/정리
    constraints = spec.get("constraints") or {}
    bounds = (constraints.get("bounds") or {}).copy()
    if bounds:
        remap = {}
        for k, v in bounds.items():
            mk = _best_feature_match(k, af, nl_query)
            if mk in aset:
                remap[mk] = v if isinstance(v, dict) else {}
            else:
                logger.warning("[feature-reconcile] drop unknown bounds key '%s' (no match)", k)
        bounds = remap

    # feature_names 보강: target_col + bounds 키 포함
    if spec.get("target_col") and spec["target_col"] not in new_fn:
        new_fn.append(spec["target_col"])
    for k in bounds.keys():
        if k not in new_fn:
            new_fn.append(k)

    spec["feature_names"] = new_fn
    if "constraints" not in spec:
        spec["constraints"] = {}
    spec["constraints"]["bounds"] = bounds

    # 최종 검증
    if spec.get("target_col") not in aset:
        raise RuntimeError(f"target_col '{spec.get('target_col')}' not in available features")

    return spec
class LLMBridge:
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://147.47.39.144:8001/v1")).rstrip("/")
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model    = model  or os.getenv("OPENAI_MODEL", "Qwen/Qwen3-14B")
        self.url      = f"{self.base_url}/chat/completions"

        # ---- 네트워크/재시도 설정(서버 안죽게) ----
        self._timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "30"))  # 30초 기본
        self._session = requests.Session()
        retry = Retry(
            total=0,                # 여기선 Session 재시도는 0 (아래에서 직접 루프 재시도)
            connect=0, read=0, status=0,
            backoff_factor=0.0,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # ---- JSON 확보 재시도(무한루프 방지용 상한/데드라인) ----
        self._max_attempts = int(os.getenv("LLM_JSON_MAX_ATTEMPTS", "12"))    # 최대 12회 시도
        self._attempt_interval_base = float(os.getenv("LLM_JSON_BACKOFF_BASE", "0.6"))  # 지수 백오프 시작
        self._overall_deadline = float(os.getenv("LLM_JSON_OVERALL_DEADLINE", "90"))    # 90초 총 데드라인
    
    # 내부 공용 POST
    def _chat(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = self._session.post(self.url, json=payload, headers=headers, timeout=self._timeout)
            r.raise_for_status()
            try:
                return r.json()
            except ValueError as e:
                # 응답이 JSON이 아님
                raise LLMBadResponse(f"Non-JSON HTTP response from LLM: {str(e)}") from e

        except requests.Timeout as e:
            # 연결 타임아웃/응답 타임아웃 모두 여기로
            raise LLMUnavailable(f"LLM timeout: {str(e)}") from e
        except requests.ConnectionError as e:
            raise LLMUnavailable(f"LLM connection error: {str(e)}") from e
        except requests.HTTPError as e:
            # 5xx는 가용성 문제, 4xx는 바디 보고 판단
            status = getattr(e.response, "status_code", None)
            if status and 500 <= status < 600:
                raise LLMUnavailable(f"LLM HTTP {status} (server error)")
            # 4xx: 모델 쪽 포맷 문제 가능 -> BadResponse
            # (너무 긴 본문은 잘라서)
            txt = (e.response.text or "")[:200] if getattr(e, "response", None) else ""
            raise LLMBadResponse(f"LLM HTTP {status}: {txt}") from e

    # 텍스트에서 가장 바깥 JSON 객체만 강제 추출
    def _extract_outer_json(self, text: str) -> Optional[dict]:
        if not text:
            return None
        # 가장 바깥 중괄호 영역을 탐색 (중첩 괄호 대응)
        stack = []
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start = i
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            # 계속 다음 후보 탐색
                            pass
        # 마지막으로 단순 스캔
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
        return None

    def _parse_spec_from_resp(self, resp: dict) -> Optional[dict]:
        """OpenAI 호환 응답에서 tool_calls > function.arguments 또는 message.content JSON 추출"""
        try:
            msg = resp["choices"][0]["message"]
        except Exception:
            return None

        # 1) function call 경로
        calls = msg.get("tool_calls") or []
        if calls:
            try:
                args_str = calls[0]["function"]["arguments"]
                return json.loads(args_str)
            except Exception:
                pass

        # 2) content 경로
        txt = msg.get("content") or ""
        spec = self._extract_outer_json(txt)
        return spec

    def _build_payload_tool(self, nl_query: str) -> dict:
        tool_schema = {
            "type": "function",
            "function": {
                "name": "build_autocontrol_spec",
                "description": "Extract AutoControl spec (target_col, setpoint, horizon, etc.) from a Korean NL query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "taskId":      {"type": "string"},
                        "acID":        {"type": "string"},
                        "sensor_name": {"type": "string"},
                        "target_col":  {"type": "string"},
                        "feature_names": {"type": "array","items":{"type":"string"}},
                        "setpoint":    {"type": "number"},
                        "horizon":     {"type": "integer","minimum": 1},
                        "constraints": {"type": "object"}
                    },
                    "required": ["target_col","setpoint","horizon"]
                }
            }
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                    "Return ONLY a function call to build_autocontrol_spec. No prose, no <think>, no extra text."},
                {"role": "user", "content": nl_query}
            ],
            "tools": [tool_schema],
            "tool_choice": {"type": "function", "function": {"name": "build_autocontrol_spec"}},
            "temperature": 0.0,
            "max_tokens": 300
        }

    def _build_payload_json_only(self, nl_query: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                    ("너는 한국어 자연어 제어 요청을 JSON으로만 변환한다. "
                     "출력은 순수 JSON 한 덩어리여야 한다. 아무 주석/설명/문장도 금지. "
                     "필수 키: target_col(string), setpoint(number), horizon(integer≥1). "
                     "선택 키: taskId, acID, sensor_name, feature_names(array of string), constraints(object).")},
                {"role": "user", "content": nl_query}
            ],
            "temperature": 0.0,
            "max_tokens": 400
        }

    def _postprocess_spec(self, spec: dict) -> dict:
        """사소한 포맷 보정 + 기본값 채우기 (taskId/acID 등)"""
        if spec is None:
            raise LLMBadResponse("Spec is None")

        # list 강제 등
        if "feature_names" in spec and isinstance(spec["feature_names"], str):
            spec["feature_names"] = [spec["feature_names"]]
        if "constraints" in spec and spec["constraints"] is None:
            spec["constraints"] = {}
        if "constraints" not in spec:
            spec["constraints"] = {}

        # taskId/acID 기본값
        import datetime as _dt
        if not spec.get("taskId"):
            spec["taskId"] = "task_" + _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        if not spec.get("acID"):
            spec["acID"] = "ac_" + _dt.datetime.utcnow().strftime("%H%M%S%f")[:10]

        return spec

    def _extract_autocontrol_spec(self, nl_query: str, available_features: Optional[List[str]] = None) -> dict:
        import json, time, logging
        logger = logging.getLogger(__name__)

        deadline = time.monotonic() + self._overall_deadline
        attempt = 0
        last_err: Optional[Exception] = None

        while attempt < self._max_attempts and time.monotonic() < deadline:
            attempt += 1
            try:
                # 1) tool-call 프롬프트
                resp = self._chat(self._build_payload_tool(nl_query))
                spec = self._parse_spec_from_resp(resp)
                if spec is not None:
                    spec = self._postprocess_spec(spec)
                    spec = self._normalize_spec_bounds(spec, available_features=available_features)
                    spec = _reconcile_and_harden_features(spec, nl_query, available_features)
                    logger.info("[_extract_autocontrol_spec] success via tool-call on attempt #%d", attempt)
                    return spec
            except (LLMUnavailable, LLMBadResponse) as e:
                last_err = e
                logger.warning("[_extract_autocontrol_spec] tool-call failed #%d: %s", attempt, e, exc_info=True)
            except Exception as e:
                last_err = e
                logger.warning("[_extract_autocontrol_spec] tool-call exception #%d: %s", attempt, e, exc_info=True)

            try:
                # 2) JSON-only 프롬프트
                resp2 = self._chat(self._build_payload_json_only(nl_query))
                spec2 = self._parse_spec_from_resp(resp2)
                if spec2 is None:
                    # content 안의 바깥 JSON 강제 추출
                    txt = (resp2.get("choices", [{}])[0].get("message", {}).get("content") or "")
                    spec2 = self._extract_outer_json(txt)
                if spec2 is not None:
                    spec2 = self._postprocess_spec(spec2)
                    spec2 = self._normalize_spec_bounds(spec2, available_features=available_features)
                    spec2 = _reconcile_and_harden_features(spec2, nl_query, available_features)
                    logger.info("[_extract_autocontrol_spec] success via json-only on attempt #%d", attempt)
                    return spec2
            except (LLMUnavailable, LLMBadResponse) as e:
                last_err = e
                logger.warning("[_extract_autocontrol_spec] json-only failed #%d: %s", attempt, e, exc_info=True)
            except Exception as e:
                last_err = e
                logger.warning("[_extract_autocontrol_spec] json-only exception #%d: %s", attempt, e, exc_info=True)

            # 지수 백오프 (최대 5초)
            sleep_s = min(self._attempt_interval_base * (2 ** (attempt - 1)), 5.0)
            logger.warning("[_extract_autocontrol_spec] retrying in %.2fs (attempt #%d/%d)", sleep_s, attempt, self._max_attempts)
            time.sleep(sleep_s)

        # 최종 실패 처리
        if isinstance(last_err, LLMUnavailable):
            raise last_err
        raise LLMBadResponse("LLM did not return valid JSON within attempts/deadline")
    def _normalize_spec_bounds(self, spec: dict, available_features: list[str] | None = None) -> dict:
        """
        - constraints.bounds 구조 강제
        - constraints 안의 {FEATURE: {min/max/step}} 형태도 bounds로 흡수
        - min_* / max_* 키도 bounds로 흡수
        - 숫자 문자열 → float, min>max 스왑
        - (available_features가 주어지면) 가용 컬럼 외 키 제거
        - 최종적으로 constraints에는 bounds만 남김
        """
        spec = dict(spec or {})
        constraints = dict((spec.get("constraints") or {}))
        bounds = dict((constraints.get("bounds") or {}))

        # 기준 피처 목록(매핑 시 사용)
        avail = list(available_features or (spec.get("feature_names") or []))

        # 1) constraints 안의 {FEATURE: {min/max}} 직접 바운드 흡수
        for k in list(constraints.keys()):
            if k == "bounds":
                continue
            v = constraints[k]
            # {FEATURE: {min/max}} 형태
            if _is_bound_dict(v):
                chosen = _best_feature_match(k, avail, "") if avail else k
                b = bounds.get(chosen) or {}
                for kk in ("min", "max", "step"):
                    if kk in v and v[kk] is not None:
                        b[kk] = v[kk]
                bounds[chosen] = b
                del constraints[k]  # 흡수 후 원래 키 제거

        # 2) min_* / max_* 편의 키 흡수
        for k in list(constraints.keys()):
            if k == "bounds":
                continue
            v = constraints[k]
            if not isinstance(k, str) or not isinstance(v, (int, float, str)):
                continue
            if k.startswith("min_") or k.startswith("max_"):
                kind = "min" if k.startswith("min_") else "max"
                raw = k[4:]
                chosen = _best_feature_match(raw, avail, "") if avail else raw
                try:
                    vv = float(v)
                except Exception:
                    logger.warning("[_normalize_spec_bounds] cannot cast %s to float: %s", k, v)
                    vv = None
                b = bounds.get(chosen) or {}
                if vv is not None:
                    b[kind] = vv
                bounds[chosen] = b
                del constraints[k]  # 흡수 후 원래 키 제거

        # 3) 타입 보정 + 스왑 + 가용 피처 필터링
        allowed = set(available_features) if available_features else None
        for feat in list(bounds.keys()):
            # 가용 컬럼 외면 제거
            if allowed and feat not in allowed:
                logger.warning("[_normalize_spec_bounds] drop unknown bound key '%s' (not available)", feat)
                del bounds[feat]
                continue
            b = bounds.get(feat) or {}
            if not isinstance(b, dict):
                b = {}
            for kk in ("min", "max", "step"):
                if kk in b and b[kk] is not None:
                    try:
                        b[kk] = float(b[kk])
                    except Exception:
                        logger.warning("[_normalize_spec_bounds] '%s.%s' not numeric: %s", feat, kk, b[kk])
                        del b[kk]
            mn, mx = b.get("min"), b.get("max")
            if isinstance(mn, (int, float)) and isinstance(mx, (int, float)) and mn > mx:
                b["min"], b["max"] = mx, mn
            bounds[feat] = b

        # 4) 최종 정리: constraints에는 bounds만 남기기
        constraints = {"bounds": bounds}
        spec["constraints"] = constraints
        logger.debug("[_normalize_spec_bounds] final bounds=%s", json.dumps(bounds, ensure_ascii=False))
        return spec

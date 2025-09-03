import os, json, requests, datetime
from typing import Any, Dict, List

class LLMBridge:
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://147.47.39.144:8001/v1")).rstrip("/")
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model    = model  or os.getenv("OPENAI_MODEL", "Qwen/Qwen3-14B")
        self.url      = f"{self.base_url}/chat/completions"

    def _chat(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(self.url, json=payload, headers=headers, timeout=60)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"LLM request failed ({r.status_code}) to {self.url}: {r.text[:200]}"
            ) from e
        return r.json()

    def _extract_autocontrol_spec(self, nl_query: str) -> dict:
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

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content":
                        "Return ONLY a function call to build_autocontrol_spec. "
                        "No prose, no <think>, no extra text."},
                    {"role": "user", "content": nl_query}
                ],
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": "build_autocontrol_spec"}},
                "temperature": 0.0,
                "max_tokens": 300
            }
            resp = self._chat(payload)
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if calls:
                args_str = calls[0]["function"]["arguments"]
                spec = json.loads(args_str)
            else:
                txt = msg.get("content") or ""
                s, e = txt.find("{"), txt.rfind("}")
                if s != -1 and e != -1:
                    spec = json.loads(txt[s:e+1])
                else:
                    raise ValueError("no tool call")
        except Exception:
            payload2 = {
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
            resp2 = self._chat(payload2)
            txt = resp2["choices"][0]["message"]["content"] or "{}"
            s, e = txt.find("{"), txt.rfind("}")
            if s == -1 or e == -1:
                raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
            spec = json.loads(txt[s:e+1])

        import datetime as _dt
        if not spec.get("taskId"):
            spec["taskId"] = "task_" + _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        if not spec.get("acID"):
            spec["acID"] = "ac_" + _dt.datetime.utcnow().strftime("%H%M%S%f")[:10]
        return spec

    def _narrate(self, payload: Dict[str, Any]) -> str:
        try:
            system_msg = (
                "너의 임무는 제조 자유제어 파이프라인의 JSON 결과를 한국어로 "
                "필드 누락 없이 그대로 설명문으로 풀어쓰는 것이다. "
                "절대 새로운 가정/수치/해석을 추가하지 말고 JSON에 있는 값만 사용하라. "
                "각 필드는 사람이 즉시 이해할 수 있도록 간결하게 항목별로 써라. "
                "출력은 순수 텍스트이며 Markdown 불릿을 사용하되 표나 코드블록은 쓰지 마라."
            )
            user_msg = (
                "다음 JSON의 모든 항목을 한국어로 자세히 설명하되, 아래 출력 형식을 지켜라.\n\n"
                "출력 형식:\n"
                "제목: PRISM 자유제어 결과 상세 보고\n"
                "\n"
                "섹션1: 요청 정보\n"
                "- taskId: ...\n"
                "- timeRange: ... (없으면 '정보 없음')\n"
                "- sensor_name / target_col(또는 target_cols): ...\n"
                "\n"
                "섹션2: 모델/자유제어\n"
                "- modelSelected: ...\n"
                "- pred_len: ...\n"
                "- prediction: 값이 많으면 앞 5개와 뒤 5개, 전체 개수 표기. (예: [앞 5] 1,2,3,4,5 / [뒤 5] ... / 총 N개)\n"
                "\n"
                "섹션3: 위험도\n"
                "- risk.riskLevel: ...\n"
                "- risk.exceedsThreshold 등 부가 필드가 있으면 모두 명시\n"
                "- suggestedActions가 있으면 모두 나열, 없으면 '정보 없음'\n"
                "\n"
                "섹션4: 변수 기여(설명)\n"
                "- explanation.importantFeatures: 이름과(있으면) 기여도, 최대 5개\n"
                "- explanation.method: ... (있으면)\n"
                "\n"
                "섹션5: 데이터/파이프라인 정보\n"
                "- csv_path, df_info(rows, cols), features_start_col_index_1based 등 데이터 관련 모든 필드 표시\n"
                "- feature_names는 개수가 많으면 첫 5개 + 총 개수 표기\n"
                "\n"
                "섹션6: 처리 이벤트 타임라인\n"
                "- events 전체를 시간순으로 한 줄씩 간결히 설명 (없으면 '정보 없음')\n"
                "\n"
                "섹션7: 결론\n"
                "- 현재 상태(정상/주의/위험 등 JSON 근거로) 한 줄\n"
                "- 바로 취해야 할 조치(있으면), 없으면 '특이 조치 없음'\n"
                "\n"
                "JSON:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "max_tokens": 1200
            }
            resp = self._chat(data)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_ko(payload)

    def _fallback_ko(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("PRISM 자유제어 결과 상세 보고 (폴백 모드)")
        lines.append("")

        def fmt_scalar(v: Any) -> str:
            if v is None:
                return "정보 없음"
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        def walk(key: str, val: Any, indent: int = 0):
            pad = "  " * indent
            bullet = "- "
            if isinstance(val, dict):
                lines.append(f"{pad}{bullet}{key}:")
                if not val:
                    lines.append(f"{pad}  (빈 객체)")
                for k, v in val.items():
                    walk(k, v, indent + 1)
            elif isinstance(val, list):
                lines.append(f"{pad}{bullet}{key}:")
                if not val:
                    lines.append(f"{pad}  (빈 리스트)")
                else:
                    if all(not isinstance(x, (dict, list)) for x in val):
                        if len(val) > 12:
                            head = ", ".join(fmt_scalar(x) for x in val[:5])
                            tail = ", ".join(fmt_scalar(x) for x in val[-5:])
                            lines.append(f"{pad}  [앞 5] {head} / [뒤 5] {tail} / 총 {len(val)}개")
                        else:
                            joined = ", ".join(fmt_scalar(x) for x in val)
                            lines.append(f"{pad}  {joined}")
                    else:
                        for i, item in enumerate(val):
                            item_key = f"{key}[{i}]"
                            walk(item_key, item, indent + 1)
            else:
                lines.append(f"{pad}{bullet}{key}: {fmt_scalar(val)}")

        top_keys = list(payload.keys())
        for first in ("data", "metadata", "code"):
            if first in top_keys:
                top_keys.remove(first)
                top_keys.insert(0, first)

        for k in top_keys:
            walk(k, payload[k], 0)

        return "\n".join(lines)

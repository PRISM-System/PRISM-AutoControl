import os
import requests
import json


class LLMBridge:
    def __init__(self,
                 base_url: str = "http://localhost:8000",
                 agent_name: str | None = "autonomous_control_agent",
                 timeout: float = 30.0,
                 verify: bool = False):
        self.base_url = base_url.rstrip("/")
        # Prism-Core agent invoke endpoint
        self.agent_name = agent_name or os.getenv("PRISM_AGENT_NAME", "autonomous_control_agent")
        self.invoke_url = f"{self.base_url}/api/agents/{self.agent_name}/invoke"
        self.timeout = timeout
        self.verify = verify

        self.session = requests.Session()
        self.user_id = None 

    # Login is not required for Prism-Core agent invocation; removed.

    def chat(self,
             prompt: str,
             system_prompt: str = "너는 산업 제어 및 공정 최적화 분야의 분석 전문가야.",
             model: str = "",
             temperature: float = 0.7,
             max_tokens: int = 2048,
             top_p: float = 1.0,
             stream: bool = False) -> str:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        # Prism-Core invoke payload
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": None,
            "use_tools": False,
            "max_tool_calls": 0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                # Optionally forward messages context if needed by server
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        resp = self.session.post(
            self.invoke_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify
        )
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"요청 실패: {resp.status_code}, {resp.text}")
            raise RuntimeError(f"LLM 요청 실패: {e}")

        data = resp.json()
        try:
            # Prism-Core invoke response shape: { text, tools_used, tool_results, metadata }
            return data.get("text", json.dumps(data, ensure_ascii=False))
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)

    def narrate(self, text: str) -> str:
        return self.chat(prompt=text)

if __name__ == "__main__":
    llm = LLMBridge(
        base_url=os.getenv("PRISM_CORE_BASE_URL", "http://localhost:8000"),
        agent_name=os.getenv("PRISM_AGENT_NAME", "autonomous_control_agent"),
        verify=False
    )

    prompt = "배고픈데 저녁 메뉴 추천해줘"
    response = llm.narrate(prompt)

    print("\n=== LLM 응답 ===")
    print(response)

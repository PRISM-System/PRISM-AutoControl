import os
import requests
import json


class LLMBridge:
    def __init__(self,
                 base_url: str = "https://grnd.bimatrix.co.kr/django/agi",
                 username: str | None = None,
                 password: str | None = None,
                 timeout: float = 30.0,
                 verify: bool = False):
        self.base_url = base_url.rstrip("/")
        self.login_url = f"{self.base_url}/api/login/"
        self.llm_url = f"{self.base_url}/llm-agent/"
        self.username = username or os.getenv("BIMATRIX_ID", "")
        self.password = password or os.getenv("BIMATRIX_PW", "")
        self.timeout = timeout
        self.verify = verify

        self.session = requests.Session()
        self.user_id = None 

    def login(self) -> dict:
        payload = {"username": self.username, "password": self.password}
        headers = {"accept": "application/json"}

        resp = self.session.post(
            self.login_url, json=payload, headers=headers,
            timeout=self.timeout, verify=self.verify
        )
        resp.raise_for_status()

        data = resp.json()
        print(f"로그인 성공: {data}")
        self.user_id = data.get("user_id")
        return data

    def chat(self,
             prompt: str,
             system_prompt: str = "너는 산업 제어 및 공정 최적화 분야의 분석 전문가야. 제어 변수, 타겟 변수, 그로 인한 제어 결과 등의 여러 제어 후보군을 전달해주면 그걸 공정 특성을 고려하여 각각의 결과를 분석해주고, 최종적인 후보를 선정해주면 돼.",
             model: str = "/root/models/openai/gpt-oss-120b",
             temperature: float = 0.7,
             max_tokens: int = 10000,
             top_p: float = 1.0,
             stream: bool = False) -> str:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        resp = self.session.post(
            self.llm_url,
            json=payload,
            headers=headers,
            verify=self.verify
        )
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"요청 실패: {resp.status_code}, {resp.text}")
            raise RuntimeError(f"LLM 요청 실패: {e}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)

    def narrate(self, text: str) -> str:
        return self.chat(prompt=text)

if __name__ == "__main__":
    llm = LLMBridge(
        username="kaist",
        password="kaist1234",
        verify=False
    )

    llm.login()

    prompt = "배고픈데 저녁 메뉴 추천해줘"
    response = llm.narrate(prompt)

    print("\n=== LLM 응답 ===")
    print(response)

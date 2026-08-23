#!/usr/bin/env python3
import json
import sys
import urllib.error
import urllib.request

from app.config import config_value


def main() -> int:
    api_key = config_value("DASHSCOPE_API_KEY") or config_value("ZHIPUAI_API_KEY")
    if not api_key:
        print("[ERROR] Missing API key. Fill DASHSCOPE_API_KEY (or ZHIPUAI_API_KEY) in .env.")
        return 1

    base_url = config_value("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    url = base_url.rstrip("/") + "/chat/completions"
    model = config_value("LLM_MODEL", "glm-5")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with only: pong"}],
        "stream": False,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"[OK] HTTP {resp.status}")
            data = json.loads(body)
            message = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            print("[MODEL]", data.get("model", ""))
            print("[REPLY]", message.strip())
            return 0
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] {e.code}")
        print(detail)
        return 2
    except Exception as e:
        print(f"[NETWORK/CLIENT ERROR] {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())

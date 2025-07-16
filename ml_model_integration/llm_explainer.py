import requests
import sys
import os

# 🔧 Add parent directory to sys.path to import config properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config


class LLMExplainer:
    def __init__(self, api_key: str = Config.GROQ_API_KEY, model: str = Config.GROQ_MODEL):
        self.api_key = api_key
        self.api_url = f"{Config.GROQ_BASE_URL}/chat/completions"
        self.model = model

    def explain(self, transaction_context: dict) -> str:
        prompt = self._generate_prompt(transaction_context)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial scam detection assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 200
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            print(f"🔍 Sending LLM request to Groq: {self.api_url}")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            print(f"↩️ Status Code: {response.status_code}")

            if response.status_code != 200:
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", {}).get("message", response.text)
                except Exception:
                    error_message = response.text
                print(f"⚠️ Non-200 response body: {error_message}")
                return f"⚠️ AI explanation unavailable: {error_message}"

            result = response.json()
            reply = result['choices'][0]['message']['content'].strip()

            if not reply:
                print(f"⚠️ Empty 'content' in response.")
                return "⚠️ AI explanation unavailable: No explanation returned."

            return reply

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Exception: {e}")
            return "⚠️ AI explanation unavailable: Network error or timeout."

        except Exception as e:
            print(f"❌ General LLM Error: {e}")
            return "⚠️ AI explanation unavailable (unexpected error)."

    def _generate_prompt(self, context: dict) -> str:
        features = []

        if context.get("urgency_level", 0) == 2:
            features.append("the transaction is marked highly urgent")
        if context.get("is_trusted_upi") == 0:
            features.append("the UPI ID is not verified or trusted")
        if context.get("is_new_merchant") == 1:
            features.append("the receiver is a new merchant")
        if context.get("match_known_scam_phrase") == 1:
            features.append("a known scam phrase is detected")
        if context.get("is_budget_exceeded") == 1:
            features.append("it exceeds your budget limit")
        if context.get("is_high_amount") == 1:
            features.append("the transaction amount is unusually high")

        reason = ", and ".join(features) if features else "no strong risk indicators were found"
        risk_level = context.get("risk_level", "UNKNOWN")

        return (
            f"The transaction has been flagged as {risk_level} risk because {reason}. "
            f"Write a short explanation that can be shown to a user."
        )

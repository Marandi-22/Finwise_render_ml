import requests
import json
import sys
import os

# 🔧 Add parent directory to sys.path to import config properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config


class LLMExplainer:
    def __init__(self, api_key: str = Config.OPENAI_API_KEY, model: str = Config.OPENAI_MODEL):
        self.api_key = api_key
        self.api_url = Config.OPENAI_BASE_URL + "/chat/completions"
        self.model = model

    def explain(self, transaction_context: dict) -> str:
        prompt = self._generate_prompt(transaction_context)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk explainer. Respond concisely and clearly."},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            print(f"🔍 Sending LLM request to: {self.api_url}")
            print(f"🔑 Model: {self.model}")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            print(f"↩️ Status Code: {response.status_code}")

            raw_text = response.text.strip()
            if response.status_code != 200:
                print(f"⚠️ Non-200 response body: {raw_text[:300]}")
                return f"⚠️ AI explanation unavailable: API error {response.status_code}."

            if not raw_text:
                print(f"⚠️ Empty response body!")
                return "⚠️ AI explanation unavailable: Empty response from LLM."

            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error: {e}")
                print(f"↩️ Raw content: {raw_text[:300]}")
                return "⚠️ AI explanation unavailable: LLM returned invalid JSON."

            # Safely extract content
            choices = response_json.get('choices')
            if not choices or not choices[0].get('message'):
                print(f"⚠️ Missing 'choices' or 'message' in response: {response_json}")
                return "⚠️ AI explanation unavailable: Incomplete response."

            reply = choices[0]['message'].get('content', '').strip()
            if not reply:
                print(f"⚠️ Empty 'content' in message.")
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

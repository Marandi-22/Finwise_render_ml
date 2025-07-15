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
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()['choices'][0]['message']['content']
            return reply.strip()

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "⚠️ AI explanation unavailable (rate-limited or server error). Please retry later."

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

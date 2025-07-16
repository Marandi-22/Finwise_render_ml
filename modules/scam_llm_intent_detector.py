import logging
import re
import time
import json
import requests
from typing import Tuple, List
from utils.cache_manager import cache_manager
from utils.validators import Validators, ValidationError
from config import Config

logger = logging.getLogger(__name__)

class ScamLLMIntentDetector:
    """LLM-enhanced scam detection via OpenRouter"""

    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.base_url = Config.OPENAI_BASE_URL or "https://openrouter.ai/api/v1/chat/completions"
        self.model = Config.OPENAI_MODEL
        self._compiled_patterns = None
        self._last_api_call = 0
        self._min_api_interval = 1.0  # seconds

    def _get_compiled_patterns(self) -> List[re.Pattern]:
        if self._compiled_patterns is None:
            scam_patterns = cache_manager.get_data(Config.SCAM_PATTERNS, default=[])
            self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in scam_patterns]
        return self._compiled_patterns

    def _check_pattern_match(self, reason_text: str) -> bool:
        return any(p.search(reason_text) for p in self._get_compiled_patterns())

    def _call_llm_with_retry(self, reason_text: str) -> Tuple[int, int]:
        current_time = time.time()
        if current_time - self._last_api_call < self._min_api_interval:
            time.sleep(self._min_api_interval - (current_time - self._last_api_call))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are a financial fraud detection AI. "
                    "Given a payment reason, determine if it likely indicates a scam. "
                    "Respond only in the following JSON format:"
                )},
                {"role": "user", "content": (
                    f"Payment reason: \"{reason_text}\"\n\n"
                    "Respond in this JSON format:\n"
                    "{\n"
                    "  \"scam\": true or false,\n"
                    "  \"urgency\": \"low\", \"medium\", or \"high\",\n"
                    "  \"explanation\": \"<brief explanation under 50 words>\"\n"
                    "}"
                )}
            ],
            "max_tokens": 150,
            "temperature": 0.3
        }

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=Config.TIMEOUT)
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"].strip()

                if reply.startswith("```json"):
                    reply = reply.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(reply)
                scam_flag = 1 if parsed.get("scam") else 0
                urgency_map = {"low": 0, "medium": 1, "high": 2}
                urgency_level = urgency_map.get(parsed.get("urgency", "").lower(), 0)

                logger.info(f"LLM ✅ parsed for: {reason_text[:40]}...")
                return scam_flag, urgency_level

            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON decode failed: {e}")
                logger.warning(f"Raw reply: {reply}")
                reply_lower = reply.lower()
                scam_flag = 1 if any(k in reply_lower for k in ["true", "scam", "fraud"]) else 0
                urgency_level = 2 if "high" in reply_lower else 1 if "medium" in reply_lower else 0
                return scam_flag, urgency_level

            except Exception as e:
                logger.warning(f"❌ LLM call attempt {attempt + 1} failed: {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    logger.error("🚨 All attempts failed.")
                    return 0, 0
                time.sleep(2 ** attempt)

        return 0, 0

    def detect_scam_intent(self, reason_text: str) -> Tuple[int, int]:
        try:
            reason_text = Validators.validate_text(reason_text, max_length=1000)
            if self._check_pattern_match(reason_text):
                logger.info(f"⚠️ Pattern matched: {reason_text[:50]}...")
                return 1, 2
            return self._call_llm_with_retry(reason_text)
        except ValidationError as e:
            logger.warning(f"⚠️ Validation failed: {e}")
            return 0, 0
        except Exception as e:
            logger.error(f"🔥 Detection failed: {e}")
            return 0, 0

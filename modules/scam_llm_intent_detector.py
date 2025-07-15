import logging
import re
import time
import json
from typing import Tuple, List
from openai import OpenAI
from utils.cache_manager import cache_manager
from utils.validators import Validators, ValidationError
from config import Config

logger = logging.getLogger(__name__)

class ScamLLMIntentDetector:
    """LLM-enhanced scam detection"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )
        self._compiled_patterns = None
        self._last_api_call = 0
        self._min_api_interval = 1.0  # Rate limiting
    
    def _get_compiled_patterns(self) -> List[re.Pattern]:
        """Get compiled regex patterns for scam detection"""
        if self._compiled_patterns is None:
            scam_patterns = cache_manager.get_data(Config.SCAM_PATTERNS, default=[])
            self._compiled_patterns = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in scam_patterns
            ]
        return self._compiled_patterns
    
    def _check_pattern_match(self, reason_text: str) -> bool:
        """Check if text matches known scam patterns"""
        patterns = self._get_compiled_patterns()
        return any(pattern.search(reason_text) for pattern in patterns)
    
    def _call_llm_with_retry(self, reason_text: str) -> Tuple[int, int]:
        """Call LLM with retry logic and rate limiting"""
        # Rate limiting
        current_time = time.time()
        if current_time - self._last_api_call < self._min_api_interval:
            time.sleep(self._min_api_interval - (current_time - self._last_api_call))
        
        system_prompt = (
            "You are a financial fraud detection AI. "
            "Given a payment reason, determine if it likely indicates a scam. "
            "Respond only in the following JSON format:"
        )
        
        user_prompt = (
            f"Payment reason: \"{reason_text}\"\n\n"
            "Respond in this JSON format:\n"
            "{\n"
            "  \"scam\": true or false,\n"
            "  \"urgency\": \"low\", \"medium\", or \"high\",\n"
            "  \"explanation\": \"<brief explanation under 50 words>\"\n"
            "}"
        )
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3,
                    timeout=Config.TIMEOUT
                )
                
                self._last_api_call = time.time()
                reply = response.choices[0].message.content.strip()
                
                # Clean LLM output before parsing
                if reply.startswith("```json"):
                    reply = reply.replace("```json", "").replace("```", "").strip()
                
                # Parse JSON response
                try:
                    parsed = json.loads(reply)
                    scam_flag = 1 if parsed.get("scam") else 0
                    
                    urgency_map = {"low": 0, "medium": 1, "high": 2}
                    urgency_level = urgency_map.get(parsed.get("urgency", "").lower(), 0)
                    
                    logger.info(f"LLM analysis completed for: {reason_text[:50]}...")
                    return scam_flag, urgency_level
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM JSON output: {e}")
                    logger.warning(f"Raw LLM response: {reply}")
                    
                    # Fallback to original parsing logic if JSON parsing fails
                    reply_lower = reply.lower()
                    scam_flag = 1 if any(word in reply_lower for word in ["true", "scam", "fraud", "suspicious"]) else 0
                    
                    if "high" in reply_lower:
                        urgency_level = 2
                    elif "medium" in reply_lower:
                        urgency_level = 1
                    else:
                        urgency_level = 0
                    
                    logger.info(f"LLM analysis completed (fallback parsing) for: {reason_text[:50]}...")
                    return scam_flag, urgency_level
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    logger.error(f"All LLM attempts failed for: {reason_text[:50]}...")
                    return 0, 0
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return 0, 0
    
    def detect_scam_intent(self, reason_text: str) -> Tuple[int, int]:
        """
        Detect scam intent in payment reason text
        
        Returns:
            Tuple[int, int]: (scam_flag, urgency_level)
                - scam_flag: 1 if scam detected, 0 otherwise
                - urgency_level: 0 (low), 1 (medium), 2 (high)
        """
        try:
            # Validate input
            reason_text = Validators.validate_text(reason_text, max_length=1000)
            
            # First check against known patterns
            if self._check_pattern_match(reason_text):
                logger.info(f"Pattern match detected for: {reason_text[:50]}...")
                return 1, 2  # High risk for known patterns
            
            # If no pattern match, use LLM analysis
            return self._call_llm_with_retry(reason_text)
            
        except ValidationError as e:
            logger.warning(f"Validation error in scam detection: {e}")
            return 0, 0
        except Exception as e:
            logger.error(f"Error in scam detection: {e}")
            return 0, 0
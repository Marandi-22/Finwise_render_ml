# ===== config.py =====
import os
from pathlib import Path
from typing import Dict, Any
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Config:
    """Centralized configuration management"""
    
    # File paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    
    TRUSTED_UPI_DB = DATA_DIR / "trusted_upi_db.json"
    USER_PAYMENT_HISTORY = DATA_DIR / "user_payment_history.json"
    BUDGET_DB = DATA_DIR / "budget_db.json"
    SCAM_PATTERNS = DATA_DIR / "scam_patterns.json"
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    OPENAI_MODEL = "openchat/openchat-7b:free"
    
    # Defaults
    DEFAULT_AMOUNT = 500.0
    DEFAULT_CATEGORY = "shopping"
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    @classmethod
    def ensure_data_dir(cls):
        """Ensure data directory exists"""
        cls.DATA_DIR.mkdir(exist_ok=True)

# ===== utils/cache_manager.py =====
import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import time

class CacheManager:
    """Thread-safe cache manager for JSON data"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._last_modified: Dict[str, float] = {}
        
    def get_data(self, file_path: Path, default: Any = None) -> Any:
        """Get cached data or load from file"""
        file_key = str(file_path)
        
        with self._lock:
            try:
                # Check if file exists
                if not file_path.exists():
                    logging.warning(f"File not found: {file_path}")
                    return default or {}
                
                # Check if cache is still valid
                current_mtime = file_path.stat().st_mtime
                if (file_key in self._cache and 
                    file_key in self._last_modified and
                    self._last_modified[file_key] == current_mtime):
                    return self._cache[file_key]
                
                # Load fresh data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update cache
                self._cache[file_key] = data
                self._last_modified[file_key] = current_mtime
                
                logging.info(f"Loaded data from {file_path}")
                return data
                
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in {file_path}: {e}")
                return default or {}
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                return default or {}

# Global cache instance
cache_manager = CacheManager()

# ===== utils/validators.py =====
import re
from typing import Optional
from datetime import datetime

class ValidationError(Exception):
    """Custom validation error"""
    pass

class Validators:
    """Input validation utilities"""
    
    UPI_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+$')
    USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    
    @staticmethod
    def validate_upi_id(upi_id: str) -> str:
        """Validate UPI ID format"""
        if not isinstance(upi_id, str):
            raise ValidationError("UPI ID must be a string")
        
        upi_id = upi_id.strip().lower()
        if not upi_id:
            raise ValidationError("UPI ID cannot be empty")
        
        if not Validators.UPI_PATTERN.match(upi_id):
            raise ValidationError("Invalid UPI ID format")
        
        return upi_id
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate user ID format"""
        if not isinstance(user_id, str):
            raise ValidationError("User ID must be a string")
        
        user_id = user_id.strip()
        if not user_id:
            raise ValidationError("User ID cannot be empty")
        
        if not Validators.USER_ID_PATTERN.match(user_id):
            raise ValidationError("Invalid User ID format")
        
        return user_id
    
    @staticmethod
    def validate_amount(amount: float) -> float:
        """Validate transaction amount"""
        if not isinstance(amount, (int, float)):
            raise ValidationError("Amount must be a number")
        
        if amount <= 0:
            raise ValidationError("Amount must be positive")
        
        if amount > 1000000:  # 10 lakh limit
            raise ValidationError("Amount too large")
        
        return float(amount)
    
    @staticmethod
    def validate_text(text: str, max_length: int = 500) -> str:
        """Validate text input"""
        if not isinstance(text, str):
            raise ValidationError("Text must be a string")
        
        text = text.strip()
        if not text:
            raise ValidationError("Text cannot be empty")
        
        if len(text) > max_length:
            raise ValidationError(f"Text too long (max {max_length} characters)")
        
        return text

# ===== 1. trusted_merchant_utils.py =====
import logging
from typing import Dict, Any
from utils.cache_manager import cache_manager
from utils.validators import Validators, ValidationError
from config import Config

logger = logging.getLogger(__name__)

class TrustedMerchantUtils:
    """Utilities for trusted merchant verification"""
    
    @staticmethod
    def is_trusted_merchant(upi_id: str) -> int:
        """Returns 1 if merchant is globally trusted, 0 otherwise"""
        try:
            # Validate input
            upi_id = Validators.validate_upi_id(upi_id)
            
            # Get trusted merchants data
            trusted_upis = cache_manager.get_data(Config.TRUSTED_UPI_DB, default=[])
            
            # Check if UPI is in trusted list
            result = 1 if upi_id in trusted_upis else 0
            logger.debug(f"Trusted merchant check for {upi_id}: {result}")
            
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in trusted merchant check: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error in trusted merchant check: {e}")
            return 0
    
    @staticmethod
    def is_new_for_user(user_id: str, upi_id: str) -> int:
        """Returns 1 if user has never transacted with this UPI ID, 0 otherwise"""
        try:
            # Validate inputs
            user_id = Validators.validate_user_id(user_id)
            upi_id = Validators.validate_upi_id(upi_id)
            
            # Get user payment history
            user_history = cache_manager.get_data(Config.USER_PAYMENT_HISTORY, default={})
            
            # Check if user has history with this UPI
            user_transactions = user_history.get(user_id, [])
            result = 0 if upi_id in user_transactions else 1
            
            logger.debug(f"New merchant check for user {user_id}, UPI {upi_id}: {result}")
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in new merchant check: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error in new merchant check: {e}")
            return 0

# ===== 2. budget_utils.py =====
import logging
from typing import Dict, Any
from utils.cache_manager import cache_manager
from utils.validators import Validators, ValidationError
from config import Config

logger = logging.getLogger(__name__)

class BudgetUtils:
    """Budget management utilities"""
    
    @staticmethod
    def get_user_budget_exceeded_flag(user_id: str, category: str, amount: float) -> int:
        """Returns 1 if amount causes budget overrun for given category, 0 otherwise"""
        try:
            # Validate inputs
            user_id = Validators.validate_user_id(user_id)
            category = Validators.validate_text(category, max_length=50).lower()
            amount = Validators.validate_amount(amount)
            
            # Get budget data
            user_budgets = cache_manager.get_data(Config.BUDGET_DB, default={})
            
            # Check if user and category exist
            if user_id not in user_budgets:
                logger.debug(f"No budget found for user {user_id}")
                return 0
            
            user_budget = user_budgets[user_id]
            if category not in user_budget:
                logger.debug(f"No budget found for category {category} for user {user_id}")
                return 0
            
            # Check if amount exceeds budget
            budget_limit = float(user_budget[category])
            result = 1 if amount > budget_limit else 0
            
            logger.debug(f"Budget check for user {user_id}, category {category}, amount {amount}: {result}")
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in budget check: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error in budget check: {e}")
            return 0

# ===== 3. scam_llm_intent_detector.py =====
import re
import logging
import time
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
            "You are a financial fraud detection AI. Analyze payment reasons to identify "
            "potential scams or high-risk situations. Be concise and specific."
        )
        
        user_prompt = (
            f"Payment reason: '{reason_text}'\n\n"
            "Respond with:\n"
            "1. Risk level: LOW, MEDIUM, or HIGH\n"
            "2. Is this likely a scam? YES or NO\n"
            "3. Brief explanation (max 50 words)"
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
                reply = response.choices[0].message.content.lower()
                
                # Parse response
                scam_flag = 1 if any(word in reply for word in ["yes", "scam", "fraud", "suspicious"]) else 0
                
                if "high" in reply:
                    urgency_level = 2
                elif "medium" in reply:
                    urgency_level = 1
                else:
                    urgency_level = 0
                
                logger.info(f"LLM analysis completed for: {reason_text[:50]}...")
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

# ===== 4. transaction_classifier_utils.py =====
import logging
from datetime import datetime
from typing import Tuple, Optional
from utils.validators import ValidationError
from config import Config

logger = logging.getLogger(__name__)

class TransactionClassifierUtils:
    """Transaction classification utilities"""
    
    @staticmethod
    def classify_time_and_day(timestamp: Optional[datetime] = None) -> Tuple[str, str]:
        """
        Classify transaction time and day
        
        Returns:
            Tuple[str, str]: (time_of_day, day_of_week)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if not isinstance(timestamp, datetime):
                raise ValidationError("Timestamp must be a datetime object")
            
            hour = timestamp.hour
            day_name = timestamp.strftime("%A").lower()
            
            # Classify time of day
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            
            logger.debug(f"Transaction classified as {time_of_day} on {day_name}")
            return time_of_day, day_name
            
        except ValidationError as e:
            logger.warning(f"Validation error in time classification: {e}")
            return "unknown", "unknown"
        except Exception as e:
            logger.error(f"Error in time classification: {e}")
            return "unknown", "unknown"

# ===== 5. amount_classifier.py =====
import re
import logging
from typing import Tuple, Dict
from utils.validators import Validators, ValidationError
from config import Config

logger = logging.getLogger(__name__)

class AmountClassifier:
    """Amount and category classification utilities"""
    
    def __init__(self):
        self.keyword_map = {
            # Education
            "college": "education", "fees": "education", "book": "education",
            "school": "education", "tuition": "education", "course": "education",
            
            # Entertainment
            "movie": "entertainment", "game": "entertainment", "fun": "entertainment",
            "concert": "entertainment", "show": "entertainment", "party": "entertainment",
            
            # Food
            "food": "food", "meal": "food", "zomato": "food", "swiggy": "food",
            "restaurant": "food", "dinner": "food", "lunch": "food", "breakfast": "food",
            
            # Health
            "hospital": "health", "medicine": "health", "clinic": "health",
            "doctor": "health", "pharmacy": "health", "medical": "health",
            
            # Shopping
            "dress": "shopping", "buy": "shopping", "purchase": "shopping",
            "amazon": "shopping", "flipkart": "shopping", "clothes": "shopping",
            
            # Transport
            "bus": "transport", "taxi": "transport", "uber": "transport",
            "ola": "transport", "metro": "transport", "train": "transport",
            
            # Utilities & Bills
            "electricity": "utilities", "water": "utilities", "bill": "bills",
            "phone": "bills", "internet": "bills", "gas": "utilities"
        }
        
        # Compiled regex patterns
        self.amount_patterns = [
            re.compile(r'(?:rs\.?|₹)\s*(\d{1,6}(?:,\d{3})*)', re.IGNORECASE),
            re.compile(r'(\d{1,6}(?:,\d{3})*)\s*(?:rs\.?|₹)', re.IGNORECASE),
            re.compile(r'(\d{1,6}(?:,\d{3})*)\s*rupees?', re.IGNORECASE)
        ]
    
    def _extract_amount(self, text: str) -> float:
        """Extract amount from text"""
        for pattern in self.amount_patterns:
            match = pattern.search(text)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return Config.DEFAULT_AMOUNT
    
    def _classify_category(self, text: str) -> str:
        """Classify transaction category based on keywords"""
        text_lower = text.lower()
        
        # Score categories based on keyword matches
        category_scores = {}
        for keyword, category in self.keyword_map.items():
            if keyword in text_lower:
                category_scores[category] = category_scores.get(category, 0) + 1
        
        # Return category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return Config.DEFAULT_CATEGORY
    
    def estimate_amount_and_category(self, reason: str) -> Tuple[float, str]:
        """
        Extract amount and estimate category from transaction reason
        
        Returns:
            Tuple[float, str]: (estimated_amount, category)
        """
        try:
            # Validate input
            reason = Validators.validate_text(reason, max_length=1000)
            
            # Extract amount
            amount = self._extract_amount(reason)
            
            # Classify category
            category = self._classify_category(reason)
            
            logger.debug(f"Amount classification: {amount}, Category: {category}")
            return amount, category
            
        except ValidationError as e:
            logger.warning(f"Validation error in amount classification: {e}")
            return Config.DEFAULT_AMOUNT, Config.DEFAULT_CATEGORY
        except Exception as e:
            logger.error(f"Error in amount classification: {e}")
            return Config.DEFAULT_AMOUNT, Config.DEFAULT_CATEGORY

    

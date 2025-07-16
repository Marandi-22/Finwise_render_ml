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
    
    # API Configuration (loaded from environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Defaults
    DEFAULT_AMOUNT = 500.0
    DEFAULT_CATEGORY = "shopping"
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    @classmethod
    def ensure_data_dir(cls):
        """Ensure data directory exists"""
        cls.DATA_DIR.mkdir(exist_ok=True)

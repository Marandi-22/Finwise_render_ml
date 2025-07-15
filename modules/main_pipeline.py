# ===== main_pipeline.py =====
import logging
import os
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass

# Setup log directory and file output
log_dir = "/content/drive/MyDrive/FinWise/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/transaction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# --- Module imports ---
from modules.trusted_merchant_utils import TrustedMerchantUtils
from modules.budget_utils import BudgetUtils
from modules.scam_llm_intent_detector import ScamLLMIntentDetector
from modules.transaction_classifier_utils import TransactionClassifierUtils
from modules.amount_classifier import AmountClassifier
from utils.validators import Validators, ValidationError
from config import Config

@dataclass
class TransactionAnalysisResult:
    user_id: str
    upi_id: str
    reason: str
    amount: float
    category: str
    time_of_day: str
    day_of_week: str
    is_trusted_merchant: int
    is_new_for_user: int
    budget_exceeded: int
    scam_flag: int
    urgency_level: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "upi_id": self.upi_id,
            "reason": self.reason,
            "amount": self.amount,
            "category": self.category,
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "is_trusted_merchant": self.is_trusted_merchant,
            "is_new_for_user": self.is_new_for_user,
            "budget_exceeded": self.budget_exceeded,
            "scam_flag": self.scam_flag,
            "urgency_level": self.urgency_level,
            "timestamp": self.timestamp.isoformat()
        }

class FinWiseTransactionPipeline:
    def __init__(self):
        self.trusted_merchant_utils = TrustedMerchantUtils()
        self.budget_utils = BudgetUtils()
        self.scam_detector = ScamLLMIntentDetector()
        self.transaction_classifier = TransactionClassifierUtils()
        self.amount_classifier = AmountClassifier()
        Config.ensure_data_dir()
        logger.info("✅ FinWise Transaction Pipeline initialized.")

    def analyze_transaction(
        self,
        user_id: str,
        upi_id: str,
        reason: str,
        amount: float = None,
        timestamp: datetime = None
    ) -> TransactionAnalysisResult:
        try:
            # --- Validation ---
            user_id = Validators.validate_user_id(user_id)
            upi_id = Validators.validate_upi_id(upi_id)
            reason = Validators.validate_text(reason, max_length=1000)
            if timestamp is None:
                timestamp = datetime.now()

            # --- Classification ---
            if amount is None:
                amount, category = self.amount_classifier.estimate_amount_and_category(reason)
            else:
                amount = Validators.validate_amount(amount)
                _, category = self.amount_classifier.estimate_amount_and_category(reason)

            time_of_day, day_of_week = self.transaction_classifier.classify_time_and_day(timestamp)
            is_trusted_merchant = self.trusted_merchant_utils.is_trusted_merchant(upi_id)
            is_new_for_user = self.trusted_merchant_utils.is_new_for_user(user_id, upi_id)
            budget_exceeded = self.budget_utils.get_user_budget_exceeded_flag(user_id, category, amount)
            scam_flag, urgency_level = self.scam_detector.detect_scam_intent(reason)

            # --- Risk Patch ---
            if category in ["education", "books", "school", "tuition"] and is_trusted_merchant:
                logger.info("⚖️ Override: Trusted + Educational transaction — lowering risk.")
                urgency_level = min(urgency_level, 1)
                scam_flag = 0

            # --- Result Object ---
            result = TransactionAnalysisResult(
                user_id=user_id,
                upi_id=upi_id,
                reason=reason,
                amount=amount,
                category=category,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                is_trusted_merchant=is_trusted_merchant,
                is_new_for_user=is_new_for_user,
                budget_exceeded=budget_exceeded,
                scam_flag=scam_flag,
                urgency_level=urgency_level,
                timestamp=timestamp
            )

            # --- DEBUG LOG ---
            logger.info("\n--- 🔍 Transaction Debug ---")
            logger.info(f"User: {user_id} | UPI: {upi_id}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Amount: ₹{amount:.2f} | Category: {category}")
            logger.info(f"Time: {time_of_day} | Day: {day_of_week}")
            logger.info(f"Trusted Merchant: {is_trusted_merchant}")
            logger.info(f"Is New For User: {is_new_for_user}")
            logger.info(f"Budget Exceeded: {budget_exceeded}")
            logger.info(f"Urgency Level: {urgency_level}")
            logger.info(f"Scam Flag (LLM): {scam_flag}")
            logger.info("📌 Risk Factors:")
            if budget_exceeded:
                logger.info("🔺 Budget exceeded.")
            if not is_trusted_merchant:
                logger.info("🔺 Not a trusted merchant.")
            if is_new_for_user:
                logger.info("🔺 New UPI for this user.")
            if urgency_level >= 2:
                logger.info("🔺 High urgency phrasing.")
            if scam_flag:
                logger.info("🔺 LLM flagged this as scam-related intent.")
            logger.info("--- End Debug ---\n")

            return result

        except ValidationError as e:
            logger.error(f"❌ Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise

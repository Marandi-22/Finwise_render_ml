import json
import datetime
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import the classes from finwise_pipeline
from finwise_pipeline import (
    Config,
    Validators,
    ValidationError,
    TrustedMerchantUtils,
    BudgetUtils,
    ScamLLMIntentDetector,
    TransactionClassifierUtils,
    AmountClassifier
)

# Configure logging
logger = logging.getLogger(__name__)

class QRImageUtils:
    """QR Image processing utilities"""
    
    @staticmethod
    def parse_qr_image(qr_image_path: str) -> Dict[str, Any]:
        """
        Parse QR code from image and extract UPI information
        
        Args:
            qr_image_path: Path to QR code image
            
        Returns:
            Dict containing UPI information
        """
        try:
            # Validate input
            if not isinstance(qr_image_path, str):
                raise ValidationError("QR image path must be a string")
            
            qr_path = Path(qr_image_path)
            if not qr_path.exists():
                raise ValidationError(f"QR image file not found: {qr_image_path}")
            
            # TODO: Implement actual QR code parsing logic
            # For now, return mock data - replace with actual QR parsing library
            logger.info(f"Parsing QR code from: {qr_image_path}")
            
            # Mock QR parsing result
            qr_info = {
                "upi_id": "merchant@paytm",
                "merchant_name": "Sample Merchant",
                "amount": None,  # May be preset in QR
                "currency": "INR",
                "transaction_ref": None
            }
            
            # Validate extracted UPI ID
            qr_info["upi_id"] = Validators.validate_upi_id(qr_info["upi_id"])
            
            logger.debug(f"QR parsing result: {qr_info}")
            return qr_info
            
        except ValidationError as e:
            logger.warning(f"Validation error in QR parsing: {e}")
            return {"upi_id": "unknown@merchant", "merchant_name": "Unknown", "amount": None, "currency": "INR", "transaction_ref": None}
        except Exception as e:
            logger.error(f"Error parsing QR code: {e}")
            return {"upi_id": "unknown@merchant", "merchant_name": "Unknown", "amount": None, "currency": "INR", "transaction_ref": None}

class TransactionConnector:
    """Main connector class for transaction processing"""
    
    def __init__(self):
        """Initialize connector with required components"""
        # Ensure data directory exists
        Config.ensure_data_dir()
        
        # Initialize components
        self.qr_utils = QRImageUtils()
        self.merchant_utils = TrustedMerchantUtils()
        self.budget_utils = BudgetUtils()
        self.scam_detector = ScamLLMIntentDetector()
        self.transaction_classifier = TransactionClassifierUtils()
        self.amount_classifier = AmountClassifier()
        
        logger.info("Transaction connector initialized")
    
    def build_transaction_input(self, qr_image_path: str, user_id: str, reason_text: str) -> Dict[str, Any]:
        """
        Build comprehensive transaction input from QR code, user context, and reason
        
        Args:
            qr_image_path: Path to QR code image
            user_id: User identifier
            reason_text: Transaction reason/description
            
        Returns:
            Dict containing all transaction features for risk assessment
        """
        try:
            # Validate inputs
            user_id = Validators.validate_user_id(user_id)
            reason_text = Validators.validate_text(reason_text, max_length=1000)
            
            logger.info(f"Building transaction input for user: {user_id}")
            
            # Step 1: QR Analysis
            qr_info = self.qr_utils.parse_qr_image(qr_image_path)
            upi_id = qr_info["upi_id"]
            
            # Step 2: UPI flags
            trusted_flag = self.merchant_utils.is_trusted_merchant(upi_id)
            new_flag = self.merchant_utils.is_new_for_user(user_id, upi_id)
            
            # Step 3: Scam intent detection (LLM)
            scam_flag, urgency = self.scam_detector.detect_scam_intent(reason_text)
            
            # Step 4: Time classification
            tod, dow = self.transaction_classifier.classify_time_and_day(datetime.datetime.now())
            
            # Step 5: Amount & Category estimation
            amount, category = self.amount_classifier.estimate_amount_and_category(reason_text)
            
            # Use QR amount if available and valid
            if qr_info.get("amount") and qr_info["amount"] > 0:
                try:
                    qr_amount = Validators.validate_amount(qr_info["amount"])
                    amount = qr_amount
                    logger.debug(f"Using QR code amount: {amount}")
                except ValidationError:
                    logger.warning("Invalid amount in QR code, using estimated amount")
            
            # Step 6: Budget flag
            budget_flag = self.budget_utils.get_user_budget_exceeded_flag(user_id, category, amount)
            
            # Final transaction input package
            transaction_input = {
                "amount": amount,
                "urgency_level": urgency,
                "is_trusted_upi": trusted_flag,
                "is_new_merchant": new_flag,
                "match_known_scam_phrase": scam_flag,
                "is_budget_exceeded": budget_flag,
                "category": category,
                "time_of_day": tod,
                "day_of_week": dow,
                # Additional context
                "upi_id": upi_id,
                "merchant_name": qr_info.get("merchant_name", "Unknown"),
                "user_id": user_id,
                "reason_text": reason_text,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Transaction input built successfully for user: {user_id}")
            logger.debug(f"Transaction features: {transaction_input}")
            
            return transaction_input
            
        except ValidationError as e:
            logger.error(f"Validation error in transaction input building: {e}")
            raise
        except Exception as e:
            logger.error(f"Error building transaction input: {e}")
            raise

def build_transaction_input(qr_image_path: str, user_id: str, reason_text: str) -> Dict[str, Any]:
    """
    Convenience function to maintain backward compatibility
    
    Args:
        qr_image_path: Path to QR code image
        user_id: User identifier
        reason_text: Transaction reason/description
        
    Returns:
        Dict containing all transaction features for risk assessment
    """
    connector = TransactionConnector()
    return connector.build_transaction_input(qr_image_path, user_id, reason_text)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test the connector
        connector = TransactionConnector()
        
        # Example transaction
        result = connector.build_transaction_input(
            qr_image_path="path/to/qr_code.jpg",
            user_id="user123",
            reason_text="Paying ₹500 for food delivery from Zomato"
        )
        
        print("Transaction Input Built Successfully:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error in example usage: {e}")
        logger.error(f"Error in example usage: {e}")
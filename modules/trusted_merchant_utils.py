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

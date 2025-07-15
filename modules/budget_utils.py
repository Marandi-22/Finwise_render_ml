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

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
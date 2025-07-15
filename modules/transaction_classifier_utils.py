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

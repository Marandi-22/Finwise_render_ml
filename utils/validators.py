import logging
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
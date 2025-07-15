import pandas as pd
import numpy as np
import json
from typing import Dict, List

class TransactionEnricher:
    """
    Standalone transaction enricher that converts raw API data to ML-ready format.
    """

    def __init__(self):
        # Define expected values (should match training data)
        self.expected_categories = ['education', 'entertainment', 'food', 'health', 'shopping', 'transport', 'utilities', 'bills']
        self.expected_times = ['afternoon', 'evening', 'morning', 'night']
        self.expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Configuration (should match training)
        self.high_amount_threshold = 1000

        # Feature columns in correct order (from training)
        self.feature_columns = [
            'amount', 'urgency_level', 'is_trusted_upi', 'is_new_merchant',
            'match_known_scam_phrase', 'is_budget_exceeded', 'is_high_amount',
            'is_weekend', 'is_night_transaction', 'risk_score',
            'category_education', 'category_entertainment', 'category_food',
            'category_health', 'category_shopping', 'category_transport',
            'category_utilities', 'time_of_day_evening', 'time_of_day_morning',
            'time_of_day_night', 'day_of_week_Monday', 'day_of_week_Saturday',
            'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday',
            'day_of_week_Wednesday'
        ]

    def validate_input(self, data: Dict) -> Dict:
        """Clean and validate input data"""
        clean_data = data.copy()

        # Required fields
        required = ['amount', 'urgency_level', 'is_trusted_upi', 'is_new_merchant',
                   'match_known_scam_phrase', 'is_budget_exceeded', 'category',
                   'time_of_day', 'day_of_week']

        for field in required:
            if field not in clean_data:
                raise ValueError(f"Missing required field: {field}")

        # Clean numeric fields
        clean_data['amount'] = float(clean_data['amount'])
        clean_data['urgency_level'] = int(clean_data['urgency_level'])

        # Validate ranges
        if clean_data['amount'] < 0:
            clean_data['amount'] = 0

        if clean_data['urgency_level'] not in [0, 1, 2]:
            clean_data['urgency_level'] = 0

        # Clean binary fields
        binary_fields = ['is_trusted_upi', 'is_new_merchant', 'match_known_scam_phrase', 'is_budget_exceeded']
        for field in binary_fields:
            clean_data[field] = 1 if clean_data[field] else 0

        # Validate categorical fields
        if clean_data['category'] not in self.expected_categories:
            clean_data['category'] = 'shopping'  # default

        if clean_data['time_of_day'] not in self.expected_times:
            clean_data['time_of_day'] = 'afternoon'  # default

        if clean_data['day_of_week'] not in self.expected_days:
            clean_data['day_of_week'] = 'Monday'  # default

        return clean_data

    def create_derived_features(self, data: Dict) -> Dict:
        """Create engineered features"""
        enriched = data.copy()

        # Derived features
        enriched['is_high_amount'] = 1 if data['amount'] > self.high_amount_threshold else 0
        enriched['is_weekend'] = 1 if data['day_of_week'] in ['Saturday', 'Sunday'] else 0
        enriched['is_night_transaction'] = 1 if data['time_of_day'] == 'night' else 0

        # Risk score (must match training logic)
        enriched['risk_score'] = (
            data['urgency_level'] * 0.3 +
            (1 - data['is_trusted_upi']) * 0.25 +
            data['is_new_merchant'] * 0.2 +
            data['match_known_scam_phrase'] * 0.25
        )

        return enriched

    def create_one_hot_features(self, data: Dict) -> Dict:
        """Create one-hot encoded features"""
        features = {}

        # Copy base and derived features
        numeric_features = ['amount', 'urgency_level', 'is_trusted_upi', 'is_new_merchant',
                           'match_known_scam_phrase', 'is_budget_exceeded', 'is_high_amount',
                           'is_weekend', 'is_night_transaction', 'risk_score']

        for feat in numeric_features:
            features[feat] = data[feat]

        # One-hot encode category
        for cat in ['education', 'entertainment', 'food', 'health', 'shopping', 'transport', 'utilities']:
            features[f'category_{cat}'] = 1 if data['category'] == cat else 0

        # One-hot encode time_of_day (excluding 'afternoon' - dropped in training)
        for tod in ['evening', 'morning', 'night']:
            features[f'time_of_day_{tod}'] = 1 if data['time_of_day'] == tod else 0

        # One-hot encode day_of_week (excluding 'Friday' - dropped in training)
        for dow in ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']:
            features[f'day_of_week_{dow}'] = 1 if data['day_of_week'] == dow else 0

        return features

    def enrich_transaction(self, raw_data: Dict) -> pd.DataFrame:
        """
        Main function: Convert raw API data to ML-ready DataFrame
        """
        try:
            # Step 1: Validate input
            clean_data = self.validate_input(raw_data)

            # Step 2: Create derived features
            enriched_data = self.create_derived_features(clean_data)

            # Step 3: Create one-hot encoded features
            final_features = self.create_one_hot_features(enriched_data)

            # Step 4: Create DataFrame with correct column order
            df = pd.DataFrame([final_features])

            # Step 5: Ensure all columns exist in correct order
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Select columns in training order
            df = df[self.feature_columns]

            return df

        except Exception as e:
            raise Exception(f"Error enriching transaction: {str(e)}")


def enrich_from_dict(input_dict: Dict) -> pd.DataFrame:
    """
    Simple wrapper function for API endpoints.
    Takes raw transaction dict and returns enriched DataFrame.
    """
    enricher = TransactionEnricher()
    return enricher.enrich_transaction(input_dict)
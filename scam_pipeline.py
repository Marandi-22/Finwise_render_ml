# Complete Scam Detection Pipeline for Google Colab
# Run this entire notebook cell by cell

# ================================
# 1. INSTALLATION & IMPORTS
# ================================
!pip install pandas numpy scikit-learn requests joblib

import pandas as pd
import numpy as np
import json
import joblib
import requests
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ================================
# 2. TRANSACTION ENRICHER CLASS
# ================================
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

            print(f"✅ Transaction enriched successfully! Shape: {df.shape}")
            return df

        except Exception as e:
            print(f"❌ Error enriching transaction: {str(e)}")
            raise

# ================================
# 3. SCAM PREDICTOR CLASS
# ================================
class ScamPredictor:
    """
    Standalone ML model predictor that takes enriched data and returns predictions.
    """

    def __init__(self, model_path: str = "scam_classifier_optimized.pkl",
                 scaler_path: str = "feature_scaler.pkl"):
        """
        Initialize predictor with trained model and scaler.
        """
        self.model = None
        self.scaler = None
        self.model_loaded = False

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
            print("✅ Model and scaler loaded successfully!")
        except FileNotFoundError:
            print("⚠️ Model files not found. Using mock predictions.")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")

    def predict_from_dataframe(self, enriched_df: pd.DataFrame) -> Dict:
        """
        Make prediction from enriched DataFrame.
        """
        if not self.model_loaded:
            return self._mock_prediction(enriched_df)

        try:
            # Scale the features
            X_scaled = self.scaler.transform(enriched_df)

            # Make predictions
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]

            # Calculate risk level
            if probability > 0.7:
                risk_level = "HIGH"
            elif probability > 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Get feature importance (for explanation)
            feature_importance = dict(zip(enriched_df.columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            result = {
                'is_scam': bool(prediction),
                'scam_probability': round(float(probability), 4),
                'risk_level': risk_level,
                'confidence': 'HIGH' if max(probability, 1-probability) > 0.8 else 'MEDIUM',
                'top_risk_factors': top_features,
                'model_version': 'RandomForest_v1.0'
            }

            print(f"🔮 Prediction: {result['risk_level']} Risk ({result['scam_probability']:.2%})")
            return result

        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return self._mock_prediction(enriched_df)

    def _mock_prediction(self, enriched_df: pd.DataFrame) -> Dict:
        """
        Generate mock prediction when model isn't available (for testing).
        """
        # Simple rule-based mock prediction
        row = enriched_df.iloc[0]

        risk_score = (
            row['urgency_level'] * 0.3 +
            (1 - row['is_trusted_upi']) * 0.25 +
            row['is_new_merchant'] * 0.2 +
            row['match_known_scam_phrase'] * 0.4 +
            row['is_high_amount'] * 0.1
        )

        probability = min(0.95, max(0.05, risk_score))
        prediction = 1 if probability > 0.5 else 0

        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"

        result = {
            'is_scam': bool(prediction),
            'scam_probability': round(float(probability), 4),
            'risk_level': risk_level,
            'confidence': 'MOCK',
            'top_risk_factors': [
                ('match_known_scam_phrase', 0.4),
                ('urgency_level', 0.3),
                ('is_trusted_upi', 0.25),
                ('is_new_merchant', 0.2),
                ('is_high_amount', 0.1)
            ],
            'model_version': 'MOCK_v1.0'
        }

        print(f"🎭 Mock Prediction: {result['risk_level']} Risk ({result['scam_probability']:.2%})")
        return result

# ================================
# 4. LLM EXPLAINER CLASS
# ================================
class LLMExplainer:
    def __init__(self, api_key: str, model: str = "mistralai/mistral-small-3.2-24b-instruct:free"):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model

    def explain(self, transaction_context: dict) -> str:
        prompt = self._generate_prompt(transaction_context)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk explainer. Respond concisely and clearly."},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()['choices'][0]['message']['content']
            return reply.strip()
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "⚠️ Unable to generate explanation."

    def _generate_prompt(self, context: dict) -> str:
        features = []

        if context.get("urgency_level", 0) == 2:
            features.append("the transaction is marked highly urgent")
        if context.get("is_trusted_upi") == 0:
            features.append("the UPI ID is not verified or trusted")
        if context.get("is_new_merchant") == 1:
            features.append("the receiver is a new merchant")
        if context.get("match_known_scam_phrase") == 1:
            features.append("a known scam phrase is detected")
        if context.get("is_budget_exceeded") == 1:
            features.append("it exceeds your budget limit")
        if context.get("is_high_amount") == 1:
            features.append("the transaction amount is unusually high")

        reason = ", and ".join(features)
        risk_level = context.get("risk_level", "UNKNOWN")

        prompt = f"The transaction has been flagged as {risk_level} risk because {reason}. Write a short explanation that can be shown to a user."
        return prompt

# ================================
# 5. PIPELINE CLASS
# ================================
class ScamDetectionPipeline:
    """
    Complete pipeline that connects enricher, predictor, and explainer.
    """

    def __init__(self, openrouter_api_key: str = None, model_path: str = None, scaler_path: str = None):
        """
        Initialize the complete pipeline.

        Args:
            openrouter_api_key: API key for OpenRouter (optional)
            model_path: Path to trained model file (optional)
            scaler_path: Path to feature scaler file (optional)
        """
        self.enricher = TransactionEnricher()
        self.predictor = ScamPredictor(model_path or "scam_classifier_optimized.pkl",
                                       scaler_path or "feature_scaler.pkl")
        self.explainer = LLMExplainer(openrouter_api_key) if openrouter_api_key else None

        print("🚀 Scam Detection Pipeline initialized!")
        if not openrouter_api_key:
            print("⚠️ No OpenRouter API key provided. Explanations will be rule-based.")

    def analyze_transaction(self, raw_transaction: Dict) -> Dict:
        """
        Complete analysis of a transaction.

        Args:
            raw_transaction: Raw transaction data from user input

        Returns:
            Complete analysis results
        """
        try:
            print("\n" + "="*50)
            print("🔍 STARTING TRANSACTION ANALYSIS")
            print("="*50)

            # Step 1: Enrich the transaction
            print("\n📊 Step 1: Enriching transaction data...")
            enriched_df = self.enricher.enrich_transaction(raw_transaction)

            # Step 2: Make prediction
            print("\n🤖 Step 2: Making ML prediction...")
            ml_prediction = self.predictor.predict_from_dataframe(enriched_df)

            # Step 3: Generate explanation
            print("\n🧠 Step 3: Generating explanation...")
            if self.explainer:
                # Combine raw data and prediction for context
                context = {**raw_transaction, **ml_prediction}
                llm_explanation = self.explainer.explain(context)
            else:
                llm_explanation = self._generate_rule_based_explanation(raw_transaction, ml_prediction)

            # Step 4: Format final result
            final_result = {
                'transaction_id': raw_transaction.get('transaction_id', 'N/A'),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'risk_assessment': {
                    'is_scam': ml_prediction['is_scam'],
                    'scam_probability': ml_prediction['scam_probability'],
                    'risk_level': ml_prediction['risk_level'],
                    'confidence': ml_prediction['confidence']
                },
                'explanation': llm_explanation,
                'technical_details': {
                    'model_version': ml_prediction['model_version'],
                    'top_risk_factors': ml_prediction['top_risk_factors'],
                    'enriched_features': enriched_df.iloc[0].to_dict()
                }
            }

            print("\n✅ Analysis complete!")
            return final_result

        except Exception as e:
            print(f"❌ Pipeline error: {str(e)}")
            return {
                'error': str(e),
                'transaction_id': raw_transaction.get('transaction_id', 'N/A'),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }

    def _generate_rule_based_explanation(self, raw_data: Dict, prediction: Dict) -> str:
        """
        Generate rule-based explanation when LLM is not available.
        """
        risk_level = prediction['risk_level']
        probability = prediction['scam_probability']

        explanation = f"This transaction has been assessed as {risk_level} risk with a {probability:.1%} probability of being a scam."

        reasons = []
        if raw_data.get('urgency_level') == 2:
            reasons.append("high urgency level")
        if raw_data.get('is_trusted_upi') == 0:
            reasons.append("unverified UPI ID")
        if raw_data.get('is_new_merchant') == 1:
            reasons.append("new merchant")
        if raw_data.get('match_known_scam_phrase') == 1:
            reasons.append("scam phrase detected")
        if raw_data.get('is_budget_exceeded') == 1:
            reasons.append("budget exceeded")
        if raw_data.get('amount', 0) > 1000:
            reasons.append("high amount")

        if reasons:
            explanation += f" Key risk factors include: {', '.join(reasons)}."

        return explanation

    def format_results(self, results: Dict) -> str:
        """
        Format results for display.
        """
        if 'error' in results:
            return f"❌ Error: {results['error']}"

        risk = results['risk_assessment']

        formatted = f"""
🚨 SCAM DETECTION RESULTS
{'='*40}
📊 Risk Level: {risk['risk_level']}
📈 Scam Probability: {risk['scam_probability']:.1%}
🎯 Confidence: {risk['confidence']}
🔍 Is Scam: {'Yes' if risk['is_scam'] else 'No'}

💡 EXPLANATION:
{results['explanation']}

🔧 Technical Details:
Model: {results['technical_details']['model_version']}
Analysis Time: {results['analysis_timestamp']}
Transaction ID: {results['transaction_id']}
"""
        return formatted

# ================================
# 6. USER INPUT FUNCTIONS
# ================================
def get_user_input() -> Dict:
    """
    Interactive function to get transaction details from user.
    """
    print("🏦 Enter Transaction Details:")
    print("-" * 30)

    transaction = {}

    # Basic details
    transaction['transaction_id'] = input("Transaction ID (optional): ") or f"TXN_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    transaction['amount'] = float(input("Amount (₹): "))

    # Risk indicators
    print("\nRisk Indicators (0=No, 1=Yes):")
    transaction['urgency_level'] = int(input("Urgency Level (0=Low, 1=Medium, 2=High): "))
    transaction['is_trusted_upi'] = int(input("Is UPI ID trusted? (0=No, 1=Yes): "))
    transaction['is_new_merchant'] = int(input("Is new merchant? (0=No, 1=Yes): "))
    transaction['match_known_scam_phrase'] = int(input("Contains scam phrases? (0=No, 1=Yes): "))
    transaction['is_budget_exceeded'] = int(input("Budget exceeded? (0=No, 1=Yes): "))

    # Categories
    print("\nTransaction Details:")
    categories = ['education', 'entertainment', 'food', 'health', 'shopping', 'transport', 'utilities', 'bills']
    print(f"Categories: {', '.join(categories)}")
    transaction['category'] = input("Category: ").lower()

    times = ['morning', 'afternoon', 'evening', 'night']
    print(f"Times: {', '.join(times)}")
    transaction['time_of_day'] = input("Time of day: ").lower()

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"Days: {', '.join(days)}")
    transaction['day_of_week'] = input("Day of week: ").title()

    return transaction

def create_sample_transaction(risk_level: str = "high") -> Dict:
    """
    Create sample transaction for testing.

    Args:
        risk_level: "high", "medium", or "low"
    """
    samples = {
        "high": {
            "transaction_id": "TXN_HIGH_RISK_001",
            "amount": 2500.0,
            "urgency_level": 2,
            "is_trusted_upi": 0,
            "is_new_merchant": 1,
            "match_known_scam_phrase": 1,
            "is_budget_exceeded": 1,
            "category": "shopping",
            "time_of_day": "night",
            "day_of_week": "Saturday"
        },
        "medium": {
            "transaction_id": "TXN_MEDIUM_RISK_001",
            "amount": 800.0,
            "urgency_level": 1,
            "is_trusted_upi": 1,
            "is_new_merchant": 1,
            "match_known_scam_phrase": 0,
            "is_budget_exceeded": 0,
            "category": "food",
            "time_of_day": "evening",
            "day_of_week": "Friday"
        },
        "low": {
            "transaction_id": "TXN_LOW_RISK_001",
            "amount": 150.0,
            "urgency_level": 0,
            "is_trusted_upi": 1,
            "is_new_merchant": 0,
            "match_known_scam_phrase": 0,
            "is_budget_exceeded": 0,
            "category": "food",
            "time_of_day": "afternoon",
            "day_of_week": "Wednesday"
        }
    }

    return samples.get(risk_level, samples["medium"])

# ================================
# 7. MAIN EXECUTION
# ================================
def main():
    """
    Main function to run the complete pipeline.
    """
    print("🚀 Welcome to Scam Detection Pipeline!")
    print("=" * 50)

    # Initialize pipeline with default API key
    OPENROUTER_API_KEY = "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"

    pipeline = ScamDetectionPipeline(openrouter_api_key=OPENROUTER_API_KEY)

    while True:
        print("\n📋 Choose an option:")
        print("1. Analyze sample high-risk transaction")
        print("2. Analyze sample medium-risk transaction")
        print("3. Analyze sample low-risk transaction")
        print("4. Enter custom transaction")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            transaction = create_sample_transaction("high")
            print(f"\n📊 Analyzing HIGH RISK sample transaction...")
        elif choice == "2":
            transaction = create_sample_transaction("medium")
            print(f"\n📊 Analyzing MEDIUM RISK sample transaction...")
        elif choice == "3":
            transaction = create_sample_transaction("low")
            print(f"\n📊 Analyzing LOW RISK sample transaction...")
        elif choice == "4":
            transaction = get_user_input()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
            continue

        # Analyze the transaction
        results = pipeline.analyze_transaction(transaction)

        # Display results
        print(pipeline.format_results(results))

        # Ask if user wants to continue
        continue_choice = input("\nAnalyze another transaction? (y/n): ").lower()
        if continue_choice != 'y':
            print("👋 Goodbye!")
            break

# ================================
# 8. QUICK TEST FUNCTION
# ================================
def quick_test(api_key: str = None):
    """
    Quick test function for development.

    Args:
        api_key: OpenRouter API key (optional)
    """
    print("🧪 Running quick test...")

    # Use default API key if not provided
    if api_key is None:
        api_key = "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"
        print("🔑 Using default API key for LLM explanations...")

    # Fallback option for users who want to use their own key
    if api_key == "prompt":
        print("\n🔑 OpenRouter API Key Setup:")
        print("• Get your free API key from: https://openrouter.ai/")
        print("• Leave empty to use rule-based explanations")
        api_key = input("Enter your OpenRouter API key (or press Enter to skip): ").strip()
        if not api_key:
            api_key = None
            print("⚠️ No API key provided. Using rule-based explanations.")

    # Initialize pipeline
    pipeline = ScamDetectionPipeline(openrouter_api_key=api_key)

    # Test with multiple risk levels
    test_cases = [
        ("high", "HIGH RISK"),
        ("medium", "MEDIUM RISK"),
        ("low", "LOW RISK")
    ]

    results = []

    for risk_level, description in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 Testing {description} Transaction")
        print(f"{'='*60}")

        test_transaction = create_sample_transaction(risk_level)
        result = pipeline.analyze_transaction(test_transaction)

        print(pipeline.format_results(result))
        results.append(result)

        # Small delay for better readability
        import time
        time.sleep(1)

    print(f"\n✅ Quick test completed! Analyzed {len(results)} transactions.")
    return results

# ================================
# 9. RUN THE PIPELINE
# ================================
if __name__ == "__main__":
    # Uncomment the line below for interactive mode
    # main()

    # Or run quick test with API key prompt
    quick_test()

print("\n🎉 Pipeline setup complete! You can now:")
print("• Run main() for interactive mode with AI explanations")
print("• Run quick_test() for a quick demonstration with AI explanations")
print("• Run quick_test('prompt') to enter your own API key")
print("• Use pipeline.analyze_transaction(your_data) directly")
print("\n✅ Default API key is configured for immediate AI-powered explanations!")
print("🔑 API key configured: sk-or-v1-...230a")
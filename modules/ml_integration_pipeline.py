# ===== modules/ml_model_integration.py =====
import pandas as pd
import numpy as np
import json
import joblib
import requests
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class TransactionEnricher:
    def __init__(self):
        self.expected_categories = ['education', 'entertainment', 'food', 'health', 'shopping', 'transport', 'utilities', 'bills']
        self.expected_times = ['afternoon', 'evening', 'morning', 'night']
        self.expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        self.high_amount_threshold = 1000

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

    def enrich_transaction(self, raw_data: Dict) -> pd.DataFrame:
        clean = self._clean_input(raw_data)
        derived = self._create_derived_features(clean)
        final = self._one_hot_encode(derived)
        df = pd.DataFrame([final])

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns]
        return df

    def _clean_input(self, data: Dict) -> Dict:
        d = data.copy()
        d['amount'] = float(d.get('amount', 0))
        d['urgency_level'] = int(d.get('urgency_level', 0))
        for f in ['is_trusted_upi', 'is_new_merchant', 'match_known_scam_phrase', 'is_budget_exceeded']:
            d[f] = 1 if d.get(f) else 0
        d['category'] = d.get('category') if d.get('category') in self.expected_categories else 'shopping'
        d['time_of_day'] = d.get('time_of_day') if d.get('time_of_day') in self.expected_times else 'afternoon'
        d['day_of_week'] = d.get('day_of_week') if d.get('day_of_week') in self.expected_days else 'Monday'
        return d

    def _create_derived_features(self, d: Dict) -> Dict:
        d['is_high_amount'] = 1 if d['amount'] > self.high_amount_threshold else 0
        d['is_weekend'] = 1 if d['day_of_week'] in ['Saturday', 'Sunday'] else 0
        d['is_night_transaction'] = 1 if d['time_of_day'] == 'night' else 0
        d['risk_score'] = (
            d['urgency_level'] * 0.3 +
            (1 - d['is_trusted_upi']) * 0.25 +
            d['is_new_merchant'] * 0.2 +
            d['match_known_scam_phrase'] * 0.25
        )
        return d

    def _one_hot_encode(self, d: Dict) -> Dict:
        features = {k: d[k] for k in [
            'amount', 'urgency_level', 'is_trusted_upi', 'is_new_merchant',
            'match_known_scam_phrase', 'is_budget_exceeded', 'is_high_amount',
            'is_weekend', 'is_night_transaction', 'risk_score'
        ]}

        for cat in self.expected_categories[:-1]:  # drop 'bills'
            features[f'category_{cat}'] = 1 if d['category'] == cat else 0

        for tod in self.expected_times:
            if tod != 'afternoon':  # drop afternoon
                features[f'time_of_day_{tod}'] = 1 if d['time_of_day'] == tod else 0

        for dow in self.expected_days:
            if dow != 'Friday':  # drop Friday
                features[f'day_of_week_{dow}'] = 1 if d['day_of_week'] == dow else 0

        return features


class ScamPredictor:
    def __init__(self, model_path="models/scam_classifier_optimized.pkl", scaler_path="models/feature_scaler.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
        except:
            self.model = None
            self.scaler = None
            self.model_loaded = False

    def predict_from_dataframe(self, df: pd.DataFrame) -> Dict:
        if not self.model_loaded:
            return {"is_scam": False, "scam_probability": 0.0, "risk_level": "UNKNOWN", "confidence": "MOCK", "model_version": "MOCK"}

        X = self.scaler.transform(df)
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0][1]

        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
        conf = "HIGH" if max(prob, 1 - prob) > 0.8 else "MEDIUM"

        return {
            "is_scam": bool(pred),
            "scam_probability": round(float(prob), 4),
            "risk_level": risk,
            "confidence": conf,
            "model_version": "RandomForest_v1.0"
        }


class LLMExplainer:
    def __init__(self, api_key, model="mistralai/mistral-small-3.2-24b-instruct:free"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def explain(self, context: dict) -> str:
        prompt = self._generate_prompt(context)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk explainer. Respond clearly."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            r = requests.post(self.url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"

    def _generate_prompt(self, ctx: dict) -> str:
        flags = []
        if ctx.get("urgency_level") == 2:
            flags.append("marked highly urgent")
        if ctx.get("is_trusted_upi") == 0:
            flags.append("unverified UPI")
        if ctx.get("is_new_merchant") == 1:
            flags.append("first-time merchant")
        if ctx.get("match_known_scam_phrase") == 1:
            flags.append("scam phrase detected")
        if ctx.get("is_budget_exceeded") == 1:
            flags.append("exceeds user’s budget")
        if ctx.get("amount", 0) > 1000:
            flags.append("high transaction amount")

        core = ", and ".join(flags) or "some risk indicators"
        risk = ctx.get("risk_level", "UNKNOWN")
        return f"This transaction is labeled {risk} risk because it is {core}. Give a clear explanation."

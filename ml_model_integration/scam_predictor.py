
import joblib
import numpy as np

class ScamPredictor:
    def __init__(self, model_path="models/scam_classifier_optimized.pkl", scaler_path="models/feature_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.model_loaded = False

        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.model_loaded = True
            print("✅ Model and scaler loaded successfully!")
        except FileNotFoundError:
            print("⚠️ Model files not found. Using mock predictions.")
        except Exception as e:
            print(f"❌ Error loading model/scaler: {e}")

    def predict_from_dataframe(self, df):
        if not self.model_loaded:
            return self._mock_prediction(df)

        try:
            X_scaled = self.scaler.transform(df)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]

            if probability > 0.7:
                risk_level = "HIGH"
            elif probability > 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            feature_importance = dict(zip(df.columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            result = {
                "is_scam": bool(prediction),
                "scam_probability": round(float(probability), 4),
                "risk_level": risk_level,
                "confidence": "HIGH" if max(probability, 1 - probability) > 0.8 else "MEDIUM",
                "top_risk_factors": top_features,
                "model_version": "RandomForest_v1.0"
            }
            return result
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._mock_prediction(df)

    def _mock_prediction(self, df):
        row = df.iloc[0]
        risk_score = (
            row["urgency_level"] * 0.3 +
            (1 - row["is_trusted_upi"]) * 0.25 +
            row["is_new_merchant"] * 0.2 +
            row["match_known_scam_phrase"] * 0.4 +
            row["is_high_amount"] * 0.1
        )

        probability = min(0.95, max(0.05, risk_score))
        prediction = 1 if probability > 0.5 else 0

        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"

        result = {
            "is_scam": bool(prediction),
            "scam_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "confidence": "MOCK",
            "top_risk_factors": [
                ("match_known_scam_phrase", 0.4),
                ("urgency_level", 0.3),
                ("is_trusted_upi", 0.25),
                ("is_new_merchant", 0.2),
                ("is_high_amount", 0.1),
            ],
            "model_version": "MOCK_v1.0"
        }
        return result

# /ml_model_integration/unified_engine.py

import logging

logger = logging.getLogger(__name__)

class UnifiedScamDetectionEngine:
    def __init__(self, pipeline, enricher, predictor, explainer=None):
        self.main_pipeline = pipeline
        self.enricher = enricher
        self.predictor = predictor
        self.explainer = explainer

    def full_analysis(self, user_id: str, upi_id: str, reason: str, amount: float = None) -> dict:
        logger.info("🚦 Running full transaction analysis...")
        try:
            # Step 1: Run FinWise Core Pipeline
            txn_result = self.main_pipeline.analyze_transaction(
                user_id=user_id,
                upi_id=upi_id,
                reason=reason,
                amount=amount
            )

            # Step 2: Convert FinWise output to ML model input
            ml_input = {
                "amount": txn_result.amount,
                "urgency_level": txn_result.urgency_level,
                "is_trusted_upi": txn_result.is_trusted_merchant,
                "is_new_merchant": txn_result.is_new_for_user,
                "match_known_scam_phrase": txn_result.scam_flag,
                "is_budget_exceeded": txn_result.budget_exceeded,
                "category": txn_result.category,
                "time_of_day": txn_result.time_of_day,
                "day_of_week": txn_result.day_of_week.title()
            }
            enriched_df = self.enricher.enrich_transaction(ml_input)

            # Step 3: Predict using ML
            ml_output = self.predictor.predict_from_dataframe(enriched_df)

            # Step 4: Generate Explanation (Safe fallback included)
            context = {**txn_result.to_dict(), **ml_output}
            try:
                explanation = self.explainer.explain(context) if self.explainer else "⚠️ No explainer API key provided."
                if not explanation or explanation.strip() == "":
                    explanation = "⚠️ AI explanation is unavailable for this transaction."
            except Exception as e:
                logger.error(f"⚠️ LLM explanation failed: {e}")
                explanation = "⚠️ AI explanation is temporarily unavailable due to rate limits. Please try again later."

            # Step 5: Return Combined Output
            return {
                "user_id": user_id,
                "upi_id": upi_id,
                "raw_reason": reason,
                "amount": amount,
                "llm_output": txn_result.to_dict(),
                "ml_prediction": ml_output,
                "explanation": explanation,
                "final_recommendation": self._get_final_recommendation(txn_result.to_dict(), ml_output)
            }

        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {
                "user_id": user_id,
                "upi_id": upi_id,
                "raw_reason": reason,
                "amount": amount,
                "error": str(e),
                "status": "FAILED"
            }

    def _get_final_recommendation(self, core_output: dict, ml_output: dict) -> str:
        """Determine final recommendation based on core pipeline and ML output."""
        if ml_output.get("scam_probability", 0) > 0.8 or core_output.get("scam_flag", 0):
            return "🚨 High Risk – Do NOT Proceed"
        elif ml_output.get("scam_probability", 0) > 0.5:
            return "⚠️ Moderate Risk – Caution Advised"
        else:
            return "✅ Low Risk – Likely Safe"

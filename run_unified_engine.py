import logging
from typing import Dict, Any
from modules.main_pipeline import FinWiseTransactionPipeline
from ml_model_integration.transaction_enricher import TransactionEnricher
from ml_model_integration.scam_predictor import ScamPredictor
from ml_model_integration.llm_explainer import LLMExplainer

logger = logging.getLogger(__name__)

class UnifiedScamDetectionEngine:
    def __init__(self, api_key: str = None):
        self.main_pipeline = FinWiseTransactionPipeline()
        self.enricher = TransactionEnricher()
        self.predictor = ScamPredictor()
        self.explainer = LLMExplainer(api_key) if api_key else None
        self.api_key = api_key
        logger.info("✅ Unified Engine initialized")
    
    def full_analysis(self, user_id: str, upi_id: str, reason: str, amount: float = None) -> Dict[str, Any]:
        """
        Perform full LLM + ML analysis and explanation.
        """
        logger.info("🚦 Running full transaction analysis...")
        
        try:
            # Step 1: Run FinWise Core Pipeline
            txn_result = self.main_pipeline.analyze_transaction(
                user_id=user_id,
                upi_id=upi_id,
                reason=reason,
                amount=amount
            )
            logger.debug(f"✅ Core pipeline completed: {txn_result}")
            
            # Step 2: Enrich for ML
            enriched_df = self.enricher.enrich_transaction(txn_result.to_dict())
            logger.debug(f"✅ Data enriched for ML: {enriched_df.shape}")
            
            # Step 3: Predict using ML
            ml_output = self.predictor.predict_from_dataframe(enriched_df)
            logger.debug(f"✅ ML prediction completed: {ml_output}")
            
            # Step 4: Add Explanation
            context = {**txn_result.to_dict(), **ml_output}
            explanation = self.explainer.explain(context) if self.explainer else "No explainer API key provided."
            logger.debug(f"✅ Explanation generated: {explanation[:50]}...")
            
            # Step 5: Combine
            combined_result = {
                "user_id": user_id,
                "upi_id": upi_id,
                "raw_reason": reason,
                "amount": amount,
                "llm_output": txn_result.to_dict(),
                "ml_prediction": ml_output,
                "explanation": explanation,
                "final_recommendation": self._get_final_recommendation(txn_result.to_dict(), ml_output)
            }
            
            logger.info("✅ Full analysis completed successfully")
            return combined_result
            
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
    
    def _get_final_recommendation(self, llm_result: dict, ml_result: dict) -> str:
        """
        Combine LLM and ML outputs into a final recommendation.
        """
        # You can customize this logic based on your needs
        ml_risk = ml_result.get("risk_level", "UNKNOWN")
        llm_risk = llm_result.get("risk_level", "UNKNOWN")  # Assuming LLM also provides risk
        
        # Log the decision process
        logger.info(f"🧠 LLM Risk: {llm_risk}, 🤖 ML Risk: {ml_risk}")
        
        if ml_risk == "HIGH" or llm_risk == "HIGH":
            return "BLOCK - High risk detected"
        elif ml_risk == "MEDIUM" or llm_risk == "MEDIUM":
            return "WARN - Medium risk, proceed with caution"
        else:
            return "ALLOW - Low risk transaction"
    
    def quick_check(self, user_id: str, upi_id: str, reason: str, amount: float = None) -> Dict[str, Any]:
        """
        Quick ML-only check without LLM explanation (faster).
        """
        logger.info("⚡ Running quick ML check...")
        
        try:
            # Step 1: Run core pipeline
            txn_result = self.main_pipeline.analyze_transaction(
                user_id=user_id,
                upi_id=upi_id,
                reason=reason,
                amount=amount
            )
            
            # Step 2: ML prediction only
            enriched_df = self.enricher.enrich_transaction(txn_result.to_dict())
            ml_output = self.predictor.predict_from_dataframe(enriched_df)
            
            return {
                "user_id": user_id,
                "upi_id": upi_id,
                "ml_prediction": ml_output,
                "explanation": "Skipped in quick check mode",
                "recommendation": self._get_final_recommendation(txn_result.to_dict(), ml_output)
            }
            
        except Exception as e:
            logger.error(f"❌ Quick check failed: {str(e)}")
            return {"error": str(e), "status": "FAILED"}
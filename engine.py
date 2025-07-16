import os
from ml_model_integration.unified_engine import UnifiedScamDetectionEngine
from modules.main_pipeline import FinWiseTransactionPipeline
from ml_model_integration.transaction_enricher import TransactionEnricher
from ml_model_integration.scam_predictor import ScamPredictor
from ml_model_integration.llm_explainer import LLMExplainer

# ✅ No hardcoded key here
pipeline = FinWiseTransactionPipeline()
enricher = TransactionEnricher()
model_path = os.path.join(os.getcwd(), "models", "scam_classifier_optimized.pkl")
predictor = ScamPredictor(model_path=model_path)
explainer = LLMExplainer()  # ✅ uses GROQ_API_KEY + GROQ_MODEL + GROQ_BASE_URL from env

# Unified Engine
engine = UnifiedScamDetectionEngine(
    pipeline=pipeline,
    enricher=enricher,
    predictor=predictor,
    explainer=explainer
)

def run_analysis(user_id: str, upi_id: str, reason: str, amount: float):
    return engine.full_analysis(
        user_id=user_id,
        upi_id=upi_id,
        reason=reason,
        amount=amount
    )

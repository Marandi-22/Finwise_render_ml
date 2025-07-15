# engine.py

import os
from ml_model_integration.unified_engine import UnifiedScamDetectionEngine
from modules.main_pipeline import FinWiseTransactionPipeline
from ml_model_integration.transaction_enricher import TransactionEnricher
from ml_model_integration.scam_predictor import ScamPredictor
from ml_model_integration.llm_explainer import LLMExplainer

# 🔐 Replace with secure key (or later fetch from env)
OPENAI_API_KEY = "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"

# Initialize components
pipeline = FinWiseTransactionPipeline()
enricher = TransactionEnricher()
model_path = os.path.join(os.getcwd(), "models", "scam_classifier_optimized.pkl")
predictor = ScamPredictor(model_path=model_path)
explainer = LLMExplainer(api_key=OPENAI_API_KEY)

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

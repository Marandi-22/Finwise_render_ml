# /tests/test_unified_engine.py

import json
import os

from config import Config
from ml_model_integration.unified_engine import UnifiedScamDetectionEngine
from modules.main_pipeline import FinWiseTransactionPipeline as FinWiseCorePipeline
from ml_model_integration.transaction_enricher import TransactionEnricher
from ml_model_integration.scam_predictor import ScamPredictor
from ml_model_integration.llm_explainer import LLMExplainer

# Initialize components
pipeline = FinWiseCorePipeline()
enricher = TransactionEnricher()
predictor = ScamPredictor(model_path="models/scam_classifier_optimized.pkl")  # Use relative path

# Load LLM API config from env or fallback to config.py
api_key = os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY)
model = os.getenv("OPENAI_MODEL", Config.OPENAI_MODEL)

explainer = LLMExplainer(api_key=api_key, model=model)

# Create the unified engine
engine = UnifiedScamDetectionEngine(
    pipeline=pipeline,
    enricher=enricher,
    predictor=predictor,
    explainer=explainer
)

# Sample test input
result = engine.full_analysis(
    user_id="user1",
    upi_id="xyz@upi",
    reason="Investing in NFT drops before mint deadline",
    amount=1200.0
)

# Print the final result in pretty format
print("\n🚨 Final Scam Detection Output:\n")
print(json.dumps(result, indent=2))

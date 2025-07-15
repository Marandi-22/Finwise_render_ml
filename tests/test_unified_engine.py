# /tests/test_unified_engine.py

import sys
import json

# Make sure the app root is in path
sys.path.append('/content/drive/MyDrive/FinWise')

from ml_model_integration.unified_engine import UnifiedScamDetectionEngine
from modules.main_pipeline import FinWiseTransactionPipeline as FinWiseCorePipeline
from ml_model_integration.transaction_enricher import TransactionEnricher
from ml_model_integration.scam_predictor import ScamPredictor
from ml_model_integration.llm_explainer import LLMExplainer

# Initialize components
pipeline = FinWiseCorePipeline()
enricher = TransactionEnricher()
predictor = ScamPredictor(model_path="/content/drive/MyDrive/FinWise/models/scam_classifier_optimized.pkl")
explainer = LLMExplainer(api_key="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")  # 🔐 Replace with your real OpenAI key

# Create the engine
engine = UnifiedScamDetectionEngine(
    pipeline=pipeline,
    enricher=enricher,
    predictor=predictor,
    explainer=explainer
)

# Run a sample test case
result = engine.full_analysis(
    user_id="user1",
    upi_id="xyz@upi",
    reason="Investing in NFT drops before mint deadline",
    amount=1200.0
)

# Pretty print the final result
print("\n🚨 Final Scam Detection Output:\n")
print(json.dumps(result, indent=2))

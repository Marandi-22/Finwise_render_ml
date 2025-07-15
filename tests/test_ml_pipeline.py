# (Paste the script you just shared)
# Test script to verify your ML pipeline integration

# Sample test data
test_transaction = {
    "amount": 1500,
    "urgency_level": 2,
    "is_trusted_upi": False,
    "is_new_merchant": True,
    "match_known_scam_phrase": True,
    "is_budget_exceeded": False,
    "category": "shopping",
    "time_of_day": "night",
    "day_of_week": "Saturday"
}

# Test the pipeline
def test_pipeline():
    from ml_model_integration import TransactionEnricher, ScamPredictor, LLMExplainer
    
    print("=== Testing Transaction Enricher ===")
    enricher = TransactionEnricher()
    try:
        enriched_df = enricher.enrich_transaction(test_transaction)
        print(f"✅ Enrichment successful. Shape: {enriched_df.shape}")
        print(f"Features: {list(enriched_df.columns)}")
        print(f"Sample values: {enriched_df.iloc[0].head()}")
    except Exception as e:
        print(f"❌ Enrichment failed: {e}")
        return False
    
    print("\n=== Testing Scam Predictor ===")
    predictor = ScamPredictor()
    try:
        prediction = predictor.predict_from_dataframe(enriched_df)
        print(f"✅ Prediction successful: {prediction}")
        
        if prediction['confidence'] == 'MOCK':
            print("⚠️  Model files not found - running in mock mode")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    print("\n=== Testing LLM Explainer ===")
    # Skip LLM test unless you have API key
    api_key = "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"  # Replace with actual key
    if api_key != "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":
        explainer = LLMExplainer(api_key)
        try:
            explanation = explainer.explain(test_transaction)
            print(f"✅ Explanation generated: {explanation[:100]}...")
        except Exception as e:
            print(f"❌ Explanation failed: {e}")
    else:
        print("⚠️  Skipping LLM test - no API key provided")
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    print(f"\n=== Pipeline Integration: {'✅ WORKING' if success else '❌ ISSUES FOUND'} ===")

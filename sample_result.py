
sample_result = {
  "user_id": "user1",
  "upi_id": "xyz@upi",
  "raw_reason": "Investing in NFT drops before mint deadline",
  "amount": 1200.0,
  "llm_output": {
    "user_id": "user1",
    "upi_id": "xyz@upi",
    "reason": "Investing in NFT drops before mint deadline",
    "amount": 1200.0,
    "category": "shopping",
    "time_of_day": "morning",
    "day_of_week": "tuesday",
    "is_trusted_merchant": 0,
    "is_new_for_user": 1,
    "budget_exceeded": 0,
    "scam_flag": 1,
    "urgency_level": 2,
    "timestamp": "2025-07-15T05:02:45.230450"
  },
  "ml_prediction": {
    "is_scam": True,
    "scam_probability": 0.99,
    "risk_level": "HIGH",
    "confidence": "HIGH",
    "top_risk_factors": [
      ["risk_score", 0.2915],
      ["match_known_scam_phrase", 0.2051],
      ["urgency_level", 0.1854],
      ["is_budget_exceeded", 0.0660],
      ["amount", 0.0585]
    ],
    "model_version": "RandomForest_v1.0"
  },
  "explanation": "Your transaction was flagged as HIGH risk due to its urgency...",
  "final_recommendation": "High Risk – Do NOT Proceed"
}

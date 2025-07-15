
if __name__ == "__main__":
    # Ensure data directory exists
    Config.ensure_data_dir()
    
    # Example usage
    try:
        # Test trusted merchant utils
        merchant_utils = TrustedMerchantUtils()
        print(f"Is trusted: {merchant_utils.is_trusted_merchant('merchant@paytm')}")
        print(f"Is new for user: {merchant_utils.is_new_for_user('user123', 'merchant@paytm')}")
        
        # Test budget utils
        budget_utils = BudgetUtils()
        print(f"Budget exceeded: {budget_utils.get_user_budget_exceeded_flag('user123', 'food', 1000)}")
        
        # Test scam detector
        scam_detector = ScamLLMIntentDetector()
        scam_flag, urgency = scam_detector.detect_scam_intent("urgent payment needed for accident")
        print(f"Scam detection: flag={scam_flag}, urgency={urgency}")
        
        # Test transaction classifier
        time_day = TransactionClassifierUtils.classify_time_and_day()
        print(f"Time classification: {time_day}")
        
        # Test amount classifier
        amount_classifier = AmountClassifier()
        amount, category = amount_classifier.estimate_amount_and_category("paying ₹500 for food delivery")
        print(f"Amount classification: {amount}, {category}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")
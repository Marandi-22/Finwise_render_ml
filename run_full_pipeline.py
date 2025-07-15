import logging
from modules.main_pipeline import FinWiseTransactionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FinWiseTransactionPipeline()

    try:
        # Sample transaction for testing
        result = pipeline.analyze_transaction(
            user_id="user123",
            upi_id="merchant@paytm",
            reason="Paying ₹500 for groceries at the supermarket",
            amount=500.0
        )

        print("\n--- Transaction Analysis Result ---")
        for k, v in result.to_dict().items():
            print(f"{k}: {v}")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")


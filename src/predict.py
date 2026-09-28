import joblib
import pandas as pd


MODEL_PATH = "models/fraud_detection_model.pkl"

# Load the trained deployment model
model = joblib.load(MODEL_PATH)


def predict_fraud(transaction):
    """
    Predict whether a transaction is fraudulent.

    Parameters:
        transaction (dict): Transaction details.

    Returns:
        dict: Fraud probability and prediction.
    """

    # Convert transaction dictionary to DataFrame
    data = pd.DataFrame([transaction])

    # Get fraud probability
    fraud_probability = model.predict_proba(data)[0][1]

    # Classification threshold
    threshold = 0.5

    # Final prediction
    prediction = int(fraud_probability >= threshold)

    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": prediction
    }


if __name__ == "__main__":

    # Example transaction
    transaction = {
        "transaction_id": 10001,
        "amount": 8500.00,
        "transaction_hour": 2,
        "merchant_category": "Travel",
        "foreign_transaction": 1,
        "location_mismatch": 1,
        "device_trust_score": 25,
        "velocity_last_24h": 8,
        "cardholder_age": 35
    }

    result = predict_fraud(transaction)

    print("Fraud Probability:",
          f"{result['fraud_probability']:.2%}")

    if result["is_fraud"] == 1:
        print("Prediction: FRAUD")
    else:
        print("Prediction: LEGITIMATE")

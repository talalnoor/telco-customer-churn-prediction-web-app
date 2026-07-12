"""
src/prediction.py
------------------
All ML-specific logic lives here, separate from the Flask web layer.

Responsibilities:
    - load_model():    load the saved sklearn Pipeline from disk
    - FIELD_OPTIONS:   the exact categories the model was trained on
                        (used to build the form dropdowns in app.py)
    - predict_churn(): turn raw HTML form data into a model prediction
"""

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------
# These match EXACTLY what the pipeline's ColumnTransformer was fit on.
# Extracted directly from telco_churn_pipeline.joblib — do not rename these
# keys unless the model is retrained, or predictions will break.

NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Used to render dropdowns in index.html. "SeniorCitizen" is shown to the
# user as Yes/No for usability, then converted to 0/1 before prediction.
FIELD_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# Full set of columns the pipeline expects, in the same order as the
# original training DataFrame (order doesn't matter to ColumnTransformer,
# but keeping it consistent makes debugging easier).
ALL_MODEL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: str):
    """Load the trained sklearn Pipeline (preprocessor + classifier) from disk."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_churn(model, form_data: dict) -> dict:
    """
    Convert raw HTML form data into a prediction.

    Parameters
    ----------
    model : sklearn.Pipeline
        The loaded preprocessing + classification pipeline.
    form_data : dict
        Raw strings straight from request.form.to_dict().

    Returns
    -------
    dict with keys: label, will_churn (bool), churn_probability (float, %),
                     stay_probability (float, %), risk_level (str)
    """
    row = {}

    # --- Numeric fields: convert string -> float, with a clear error message ---
    for field in NUMERIC_FIELDS:
        raw_value = form_data.get(field)
        if raw_value is None or raw_value.strip() == "":
            raise ValueError(f"'{field}' is required.")
        try:
            row[field] = float(raw_value)
        except ValueError:
            raise ValueError(f"'{field}' must be a number, got '{raw_value}'.")

    # --- Categorical fields: validate against the known training categories ---
    for field, allowed_values in FIELD_OPTIONS.items():
        raw_value = form_data.get(field)
        if raw_value is None or raw_value == "":
            raise ValueError(f"'{field}' is required.")

        if field == "SeniorCitizen":
            # Form shows Yes/No, model expects 0/1
            if raw_value not in ("Yes", "No"):
                raise ValueError(f"'{field}' must be 'Yes' or 'No'.")
            row[field] = 1 if raw_value == "Yes" else 0
        else:
            if raw_value not in allowed_values:
                raise ValueError(f"'{field}' has an invalid value: '{raw_value}'.")
            row[field] = raw_value

    # --- Build a single-row DataFrame with the exact expected columns ---
    input_df = pd.DataFrame([row], columns=ALL_MODEL_COLUMNS)

    # --- Predict ---
    prediction = model.predict(input_df)[0]          # 0 = stay, 1 = churn
    probabilities = model.predict_proba(input_df)[0]  # [P(stay), P(churn)]

    stay_probability = round(float(probabilities[0]) * 100, 2)
    churn_probability = round(float(probabilities[1]) * 100, 2)

    will_churn = bool(prediction == 1)

    # Simple risk banding for a nicer UI (not part of the model itself)
    if churn_probability >= 70:
        risk_level = "High Risk"
    elif churn_probability >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "label": "Customer will churn" if will_churn else "Customer will stay",
        "will_churn": will_churn,
        "churn_probability": churn_probability,
        "stay_probability": stay_probability,
        "risk_level": risk_level,
    }
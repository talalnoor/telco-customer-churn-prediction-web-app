"""
app.py
------
Flask web application for Telco Customer Churn Prediction.

This file only handles the WEB layer:
    - loading the trained pipeline once at startup
    - defining routes (GET /, POST /predict)
    - passing form data to src/prediction.py and rendering the result

All ML logic (building the input row, calling the model, formatting
probabilities) lives in src/prediction.py to keep this file simple
and easy to maintain.
"""

import os
import logging
from flask import Flask, render_template, request

from src.prediction import load_model, predict_churn, FIELD_OPTIONS

# ---------------------------------------------------------------------------
# App + logging setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline ONCE when the app starts (not per-request).
# If this fails, we want the app to crash immediately with a clear error,
# rather than fail silently later on the first prediction request.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "telco_churn_pipeline.joblib")

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Render the empty input form."""
    return render_template("index.html", field_options=FIELD_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Read the submitted form, run it through the pipeline, and show the result.
    Any bad/missing input is caught and shown as a friendly error instead of
    crashing the app.
    """
    try:
        form_data = request.form.to_dict()
        result = predict_churn(model, form_data)
        return render_template("result.html", result=result, inputs=form_data)

    except (ValueError, KeyError) as e:
        # Expected, "our fault or user's fault" errors: bad/missing input
        logger.warning("Invalid input on /predict: %s", e)
        return render_template(
            "index.html",
            field_options=FIELD_OPTIONS,
            error=f"Please check your input: {e}",
        )

    except Exception as e:
        # Anything unexpected — log it, but still don't show a raw traceback
        logger.error("Unexpected error during prediction: %s", e)
        return render_template(
            "index.html",
            field_options=FIELD_OPTIONS,
            error="Something went wrong while making the prediction. Please try again.",
        )


# ---------------------------------------------------------------------------
# Entry point (local development only — Render/PythonAnywhere use their own
# WSGI entry points, covered in the Deployment step).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False") == "True"
    app.run(debug=debug_mode)
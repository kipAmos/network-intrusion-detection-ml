import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Classify network-flow records using the saved model.")
    parser.add_argument("--input", required=True, help="CSV file containing flow features.")
    parser.add_argument("--output", default="reports/predictions.csv")
    parser.add_argument("--model", default="models/best_intrusion_model.joblib")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run train.py first.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]
    encoder = saved["label_encoder"]
    expected_columns = saved["feature_columns"]

    data = pd.read_csv(input_path, low_memory=False)
    data.columns = [str(c).replace("\ufeff", "").strip() for c in data.columns]
    data = data.replace([np.inf, -np.inf], np.nan)

    # Add absent training columns as missing and discard unrelated extra columns.
    for column in expected_columns:
        if column not in data.columns:
            data[column] = np.nan
    X = data[expected_columns]

    predicted_ids = pipeline.predict(X)
    predicted_labels = encoder.inverse_transform(predicted_ids.astype(int))
    result = data.copy()
    result["predicted_class"] = predicted_labels

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        result["prediction_confidence"] = probabilities.max(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Model: {saved['model_name']}")
    print(f"Classified {len(result)} records.")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

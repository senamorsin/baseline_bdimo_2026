from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from preprocess import process_description

MODEL_PATH = "model.joblib"
INPUT_PATH = "test.tsv"
OUTPUT_PATH = "predictions.tsv"


def preprocess_frame(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    for col in ["vendor_name", "vendor_code", "title", "description", "shop_category_name"]:
        if col not in prepared.columns:
            prepared[col] = ""
        prepared[col] = prepared[col].fillna("").astype(str).str.lower()

    parsed_description = process_description(prepared[["description"]].copy())
    prepared["parsed_description"] = parsed_description["description"].fillna("")

    for col in numeric_columns:
        if col in parsed_description.columns:
            prepared[col] = parsed_description[col].to_numpy()
        else:
            prepared[col] = 0.0

    prepared["text"] = (
        prepared["title"]
        + " "
        + prepared["description"]
        + " "
        + prepared["parsed_description"]
        + " "
        + prepared["shop_category_name"]
        + " "
        + prepared["vendor_name"]
        + " "
        + prepared["vendor_code"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return prepared


def build_features(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    text_x = artifacts["word_vectorizer"].transform(df["text"])
    reduced_text = artifacts["svd"].transform(text_x).astype(np.float32)

    numeric_columns: list[str] = artifacts["numeric_columns"]
    if numeric_columns:
        numeric_x = (
            df.reindex(columns=numeric_columns, fill_value=0.0)
            .astype(np.float32)
            .to_numpy()
        )
        return np.hstack([reduced_text, numeric_x])

    return reduced_text


def main() -> None:
    artifacts = joblib.load(MODEL_PATH)
    numeric_columns: list[str] = artifacts["numeric_columns"]

    test_df = pd.read_csv(INPUT_PATH, sep="\t")
    prepared = preprocess_frame(test_df, numeric_columns)
    x_test = build_features(prepared, artifacts)

    dept_pred = artifacts["department_model"].predict(x_test)
    cat_pred = artifacts["category_model"].predict(x_test)

    out = pd.DataFrame(
        {
            "predicted_department_id": dept_pred.astype(int),
            "predicted_category_id": cat_pred.astype(int),
        }
    )
    out.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Saved {OUTPUT_PATH} with {len(out)} rows")


if __name__ == "__main__":
    main()

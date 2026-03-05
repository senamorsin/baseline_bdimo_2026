import joblib
import pandas as pd

from fit import (
    FEATURE_COLUMNS,
    preprocess_frame as fit_preprocess_frame,
    predict_with_artifacts,
)

MODEL_PATH = "model.joblib"
INPUT_PATH = "test.tsv"
OUTPUT_PATH = "prediction.csv"


def read_input_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[FEATURE_COLUMNS]


def preprocess_frame(
    df: pd.DataFrame, description_stopwords: set[str] | None = None
) -> pd.DataFrame:
    prepared, _ = fit_preprocess_frame(df, description_stopwords=description_stopwords)
    return prepared


def predict(input_path: str, output_path: str, model_path: str) -> None:
    artifacts = joblib.load(model_path)
    test_df = read_input_frame(input_path)

    stopwords = set(artifacts["description_stopwords"])
    prepared = preprocess_frame(test_df, description_stopwords=stopwords)
    cat_pred, dept_pred = predict_with_artifacts(prepared, artifacts)

    out = pd.DataFrame(
        {
            "department_id": dept_pred.astype(int),
            "category_id": cat_pred.astype(int),
        }
    )
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    predict(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)

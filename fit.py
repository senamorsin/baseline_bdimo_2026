from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from preprocess import process_description

RANDOM_STATE = 42
MODEL_PATH = "model.joblib"
WORD_MAX_FEATURES = 40_000
SVD_COMPONENTS = 96


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # User requested "weighed"; sklearn uses "weighted".
    return f1_score(y_true, y_pred, average="weighted")


def preprocess_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = df.copy()
    for col in ["vendor_name", "vendor_code", "title", "description", "shop_category_name"]:
        if col not in prepared.columns:
            prepared[col] = ""
        prepared[col] = prepared[col].fillna("").astype(str).str.lower()

    parsed_description = process_description(prepared[["description"]].copy())
    prepared["parsed_description"] = parsed_description["description"].fillna("")

    numeric_columns = [
        c
        for c in parsed_description.columns
        if c != "description" and pd.api.types.is_numeric_dtype(parsed_description[c])
    ]
    for col in numeric_columns:
        prepared[col] = parsed_description[col].to_numpy()

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

    return prepared, numeric_columns


def build_text_features(
    df: pd.DataFrame,
    fit: bool,
    word_vectorizer: TfidfVectorizer | None = None,
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    if fit:
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_features=WORD_MAX_FEATURES,
            sublinear_tf=True,
        )

    assert word_vectorizer is not None

    text_x = word_vectorizer.fit_transform(df["text"]) if fit else word_vectorizer.transform(df["text"])
    return text_x, word_vectorizer


def build_dense_features(
    df: pd.DataFrame,
    text_x: sparse.csr_matrix,
    fit: bool,
    numeric_columns: list[str],
    svd: TruncatedSVD | None = None,
) -> tuple[np.ndarray, TruncatedSVD]:
    if fit:
        max_components = min(SVD_COMPONENTS, text_x.shape[0] - 1, text_x.shape[1] - 1)
        n_components = max(2, max_components)
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        reduced_text = svd.fit_transform(text_x)
    else:
        assert svd is not None
        reduced_text = svd.transform(text_x)

    if numeric_columns:
        numeric_x = (
            df.reindex(columns=numeric_columns, fill_value=0.0)
            .astype(np.float32)
            .to_numpy()
        )
        full_x = np.hstack([reduced_text.astype(np.float32), numeric_x])
    else:
        full_x = reduced_text.astype(np.float32)

    assert svd is not None
    return full_x, svd


def build_department_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=40,
        max_depth=16,
        min_samples_leaf=4,
        max_samples=0.75,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def build_category_model() -> SGDClassifier:
    return SGDClassifier(
        loss="modified_huber",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


def evaluate_once(df: pd.DataFrame) -> None:
    print("Running validation split...", flush=True)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["department_id"],
    )

    train_prepared, train_numeric_cols = preprocess_frame(train_df)
    val_prepared, _ = preprocess_frame(val_df)

    train_text_x, word_vec = build_text_features(train_prepared, fit=True)
    val_text_x, _ = build_text_features(
        val_prepared,
        fit=False,
        word_vectorizer=word_vec,
    )

    x_train, svd = build_dense_features(
        train_prepared,
        train_text_x,
        fit=True,
        numeric_columns=train_numeric_cols,
    )
    x_val, _ = build_dense_features(
        val_prepared,
        val_text_x,
        fit=False,
        numeric_columns=train_numeric_cols,
        svd=svd,
    )

    y_train_dept = train_df["department_id"].to_numpy()
    y_train_cat = train_df["category_id"].to_numpy()
    y_val_dept = val_df["department_id"].to_numpy()
    y_val_cat = val_df["category_id"].to_numpy()

    dept_model = build_department_model()
    cat_model = build_category_model()
    print("Fitting validation department model...", flush=True)
    dept_model.fit(x_train, y_train_dept)
    print("Fitting validation category model...", flush=True)
    cat_model.fit(x_train, y_train_cat)

    dept_pred = dept_model.predict(x_val)
    cat_pred = cat_model.predict(x_val)

    print(f"Validation department weighted F1: {weighted_f1(y_val_dept, dept_pred):.4f}", flush=True)
    print(f"Validation category weighted F1:   {weighted_f1(y_val_cat, cat_pred):.4f}", flush=True)


def train_and_save(df: pd.DataFrame) -> None:
    print("Fitting final models on full train.tsv...", flush=True)
    prepared, numeric_cols = preprocess_frame(df)
    text_x, word_vec = build_text_features(prepared, fit=True)
    full_x, svd = build_dense_features(
        prepared,
        text_x,
        fit=True,
        numeric_columns=numeric_cols,
    )

    y_dept = df["department_id"].to_numpy()
    y_cat = df["category_id"].to_numpy()

    dept_model = build_department_model()
    cat_model = build_category_model()
    print("Fitting final department model...", flush=True)
    dept_model.fit(full_x, y_dept)
    print("Fitting final category model...", flush=True)
    cat_model.fit(full_x, y_cat)

    artifacts = {
        "word_vectorizer": word_vec,
        "svd": svd,
        "numeric_columns": numeric_cols,
        "department_model": dept_model,
        "category_model": cat_model,
    }
    joblib.dump(artifacts, MODEL_PATH, compress=3)
    print(f"Saved model artifacts to {MODEL_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run holdout validation and print weighted F1 before final training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run holdout validation only, without final full-data training.",
    )
    args = parser.parse_args()

    df = pd.read_csv("train.tsv", sep="\t")
    print(f"Loaded train.tsv with {len(df)} rows", flush=True)

    if args.eval or args.eval_only:
        evaluate_once(df)
    if not args.eval_only:
        train_and_save(df)


if __name__ == "__main__":
    main()

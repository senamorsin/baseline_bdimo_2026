import re

import joblib
import pandas as pd
from scipy.sparse import hstack
from fit import clean_text, TEXT_COLS, build_department_text, build_category_text
TEXT_COLS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]


def  predict():
    test = pd.read_csv("test.tsv", sep="\t")
    for col in TEXT_COLS:
        test[col] = test[col].fillna("")

    model = joblib.load("model.joblib")

    dep_text = build_department_text(test)
    X_dep_word = model["dep_word_vectorizer"].transform(dep_text)
    X_dep_char = model["dep_char_vectorizer"].transform(dep_text)
    X_dep = hstack([X_dep_word, X_dep_char], format="csr")
    department_pred = model["department_model"].predict(X_dep)

    cat_text = build_category_text(test)
    X_cat_word = model["cat_word_vectorizer"].transform(cat_text)
    X_cat_char = model["cat_char_vectorizer"].transform(cat_text)
    X_cat = hstack([X_cat_word, X_cat_char], format="csr")

    neighbor_idx = model["category_model"].kneighbors(X_cat, return_distance=False).ravel()
    category_pred = model["category_targets"][neighbor_idx].copy()

    title_to_category = model["title_to_category"]
    titles = test["title"].astype(str).to_numpy()
    for i, title in enumerate(titles):
        mapped = title_to_category.get(title)
        if mapped is not None:
            category_pred[i] = mapped

    submission = pd.DataFrame(
        {
            "category_id": category_pred,
            "department_id": department_pred,
        }
    )
    submission.to_csv("prediction.csv", index=False)

if __name__ == "__main__":
    predict()
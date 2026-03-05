import os
import re
import html

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

TEXT_COLS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]


def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_department_text(df):
    return (
        df["title"].map(clean_text)
        + " "
        + df["title"].map(clean_text)
        + " "
        + df["shop_category_name"].map(clean_text)
        + " "
        + df["shop_category_name"].map(clean_text)
        + " "
        + df["vendor_name"].map(clean_text)
        + " "
        + df["vendor_code"].map(clean_text)
        + " "
        + df["description"].map(clean_text)
    )


def build_category_text(df):
    return (
        df["title"].astype(str)
        + " "
        + df["title"].astype(str)
        + " "
        + df["shop_category_name"].astype(str)
        + " "
        + df["shop_category_name"].astype(str)
        + " "
        + df["vendor_name"].astype(str)
        + " "
        + df["vendor_code"].astype(str)
    )


def build_unique_title_map(df):
    title_class_counts = df.groupby("title")["category_id"].nunique()
    unique_titles = set(title_class_counts[title_class_counts == 1].index)
    title_map = (
        df[df["title"].isin(unique_titles)]
        .drop_duplicates(subset=["title", "category_id"])
        .drop_duplicates(subset=["title"])
        .set_index("title")["category_id"]
        .to_dict()
    )
    return title_map

def fit():
    train = pd.read_csv("train.tsv", sep="\t")
    for col in TEXT_COLS:
        train[col] = train[col].fillna("")

    department_text = build_department_text(train)

    dep_word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=120000,
    )
    dep_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=120000,
    )

    X_dep_word = dep_word_vectorizer.fit_transform(department_text)
    X_dep_char = dep_char_vectorizer.fit_transform(department_text)
    X_dep = hstack([X_dep_word, X_dep_char], format="csr")

    department_model = LinearSVC(C=1.4, class_weight="balanced", random_state=42)
    department_model.fit(X_dep, train["department_id"])

    category_text = build_category_text(train)

    cat_word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=120000,
    )
    cat_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=120000,
    )

    X_cat_word = cat_word_vectorizer.fit_transform(category_text)
    X_cat_char = cat_char_vectorizer.fit_transform(category_text)
    X_cat = hstack([X_cat_word, X_cat_char], format="csr")

    category_model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    category_model.fit(X_cat)

    model = {
        "dep_word_vectorizer": dep_word_vectorizer,
        "dep_char_vectorizer": dep_char_vectorizer,
        "department_model": department_model,
        "cat_word_vectorizer": cat_word_vectorizer,
        "cat_char_vectorizer": cat_char_vectorizer,
        "category_model": category_model,
        "category_targets": train["category_id"].to_numpy(),
        "title_to_category": build_unique_title_map(train),
    }

    joblib.dump(model, "model.joblib", compress=3)

    model_size_mb = os.path.getsize("model.joblib") / (1024 * 1024)
    print(f"Saved model.joblib size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    fit()
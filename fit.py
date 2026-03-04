import pandas as pd
from pipeline import preprocess_frame, extract_frame
from preprocess import format_shop_category_name, format_vendor_name
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import SGDClassifier
import pickle

RANDOM_STATE = 42


def fit():
    df = pd.read_csv("train.tsv", sep="\t")
    X, y_dep, y_cat = (
        df.drop(["department_id", "category_id"]),
        df["department_id"],
        df["category_id"],
    )

    top_vendor_names = (
        format_vendor_name(X)["vendor_name"].value_counts().head(11).index.to_list()
    )
    top_category_names = (
        format_shop_category_name(X)["shop_category_name"]
        .value_counts()
        .head(21)
        .index.to_list()
    )

    X = preprocess_frame(X, top_vendor_names, top_category_names)

    char_tfidf = TfidfVectorizer(analyzer="char").fit(X["description"])
    word_tfidf = TfidfVectorizer(analyzer="word").fit(X["description"])
    word_tfidf_svd = TruncatedSVD(256, random_state=RANDOM_STATE).fit(
        word_tfidf.transform(X["description"])
    )
    char_count = CountVectorizer(analyzer="char").fit(X["description"])
    word_count = CountVectorizer(analyzer="word").fit(X["description"])
    word_count_svd = TruncatedSVD(256, random_state=RANDOM_STATE).fit(
        word_count.transform(X["description"])
    )

    description_extractors = {
        "char_tfidf": (char_tfidf,),
        "word_tfidf": (word_tfidf, word_tfidf_svd),
        "char_count": (char_count,),
        "word_count": (word_count, word_count_svd),
    }

    vendor_name_oh = OneHotEncoder().fit(X[["vendor_name"]])
    shop_category_name_oh = OneHotEncoder().fit(X[["shop_category_name"]])

    X = extract_frame(X, description_extractors, vendor_name_oh, shop_category_name_oh)
    X = X.select_dtypes(include=np.number)
    
    cols_to_scale = [col for col in X.columns if not np.array_equal(np.sort(X[col].astype(int).unique()), np.array([0., 1.]))]
    scaler = StandardScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    model_dep = RandomForestClassifier(
        **{'n_estimators': 160,
            'max_depth': 18,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
    )
    model_dep.fit(X, y_dep)

    model_cat = SGDClassifier()
    model_cat.fit(pd.concat([X, y_dep], axis=1), y_cat)

    artifacts = {
        "top_vendor_names": top_vendor_names,
        "top_category_names": top_category_names,
        "description_extractors": description_extractors,
        "vendor_name_oh": vendor_name_oh,
        "shop_category_name_oh": shop_category_name_oh,
        "cols_to_scale": cols_to_scale,
        "scaler": scaler,
        "model_dep": model_dep,
        "model_cat": model_cat,
    }

    with open("artifacts.pkl", "rw") as f:
        pickle.dump(artifacts, f)

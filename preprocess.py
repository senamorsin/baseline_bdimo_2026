import re
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import html
from sklearn.feature_extraction.text import CountVectorizer

def parse_description(text: str):
    if pd.isna(text):
        text = ''
    text = str(text)

    features: dict = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'has_hashtag': 1 if '#' in text else 0,
        'has_url': 1 if 'http' in text.lower() or 'www.' in text.lower() else 0,
        'has_email': 1 if '@' in text else 0,
        'has_phone': 1 if bool(re.search(r'\d[\d\s\-]{9,10}', text)) else 0,
        'has_emoji': 1 if bool(re.search(r'[^\w\s]', text)) else 0,
        'has_html': 1 if '<' in text and '>' in text else 0,
        'has_linebreak': 1 if '<br>' in text.lower() or '\n' in text.lower() else 0,
        'has_list': 1 if '<ul>' in text.lower() or '<li>' in text.lower() else 0,
        'has_par': 1 if '<p>' in text.lower() else 0,
        'has_bold': 1 if '<b>' in text.lower() else 0,
        'has_div': 1 if '<div>' in text.lower() else 0,
        'has_entity': 1 if bool(re.search(r'&[a-zA-Z0-9#]+;?', text)) else 0,
        'has_header': 1 if bool(re.search(r'<h[1-6]\b[^>]*>.*?</h[1-6]>', text, re.I)) else 0,
        'has_number': 1 if bool(re.search(r'\d', text)) else 0,
        'has_currency': 1 if bool(re.search(r'[₽€£]|(руб|р\.|дол|евро)', text, re.I)) else 0,
        'has_price': 1 if bool(re.search(r'\d+\s?(₽|руб|р\.|дол|евро)', text, re.I)) else 0,
        'has_address': 1 if bool(re.search(r'(ул\.|улица|проспект|пр-т|пл\.|переулок|пер\.)', text, re.I)) else 0,
        'has_date': 1 if bool(re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', text)) else 0,
        'has_time': 1 if bool(re.search(r'\d{1,2}:\d{2}', text)) else 0,
        'has_caps': 1 if bool(re.search(r'[А-ЯЁ]{2,}', text)) else 0,
        'has_mention': 1 if '@' in text else 0,
        'has_question': 1 if '?' in text else 0,
        'has_exclamation': 1 if '!' in text else 0,
        'has_ellipsis': 1 if '...' in text or '…' in text else 0,
        'has_quotes': 1 if '"' in text or '«' in text else 0,
        'hashtag_count': len(re.findall(r'#\w+', text)),
        'url_count': len(re.findall(r'http[s]?://\S+|www\.\S+', text)),
        'number_count': len(re.findall(r'\d+', text)),
        'caps_letter_count': len(re.findall(r'[A-Z]', text)),
        'caps_word_count': len(re.findall(r'[А-ЯЁ]{2,}', text))
    }


    features['caps_letter_ratio'] = 0 if features["text_length"] == 0 else (features['caps_letter_count'] / features["text_length"])
    features['caps_word_ratio'] = 0 if features["word_count"] == 0 else (features['caps_word_count'] / features["word_count"])

    return features

def format_description(df: pd.DataFrame):
    df["description"] = (
        df["description"].str.lower()
        .replace(r"<[^>]+>", "", regex=True)
        .replace(r"[\n\r\t]+", " ", regex=True)
        .replace(r"[^a-zA-Zа-яА-Я0-9\s.,;:!?()\-\+\/_\"']", "", regex=True)
        .replace(r"\s+", " ", regex=True)
    )
    df.loc[df["description"] == " ", "description"] = ""
    return df


def process_description(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df["description"] = new_df["description"].fillna("").apply(html.unescape)
    features_df = pd.DataFrame(new_df["description"].apply(parse_description).to_list())
    features_df.index = new_df.index
    return format_description(new_df.join(features_df)).sort_index()


def extract_description(df: pd.DataFrame, vectorizers: dict[str, CountVectorizer]) -> pd.DataFrame:
    dfs = []
    for name, vectorizer in vectorizers.items():
        sparse = vectorizer.transform(df["description"])
        cols = [f"{name}_{i}" for i in range(sparse.shape[1])]
        sparse_df = pd.DataFrame(sparse.toarray(), columns=cols) # type: ignore
        dfs.append(sparse_df.sort_index())
    return df.sort_index().join(dfs, how="left")


def fromat_vendor_name(df: pd.DataFrame) -> pd.DataFrame:
    df["vendor_name"] = df["vendor_name"].str.lower()
    return df


def process_vendor_name(df: pd.DataFrame, top_cats: list[str]) -> pd.DataFrame:
    new_df = df.copy()
    new_df = fromat_vendor_name(new_df)
    new_df.loc[new_df["vendor_name"].isin(("no brand", "без бренда")), "vendor_name"] = "нет бренда"
    new_df.loc[~new_df["vendor_name"].isin(top_cats), "vendor_name"] = "OTHER"
    return new_df


def format_shop_category_name(df: pd.DataFrame):
    df["shop_category_name"] = (
        df["shop_category_name"].str.lower()
        .replace(r"[^a-zA-Zа-яА-Я]", "", regex=True)
        .replace(r"\s+", " ", regex=True)
    )
    df.loc[df["shop_category_name"] == " ", "shop_category_name"] = ""
    return df


def process_shop_category_name(df: pd.DataFrame, top_cats: list[str]):
    new_df = df.copy()
    new_df["shop_category_name"] = new_df["shop_category_name"].fillna("")
    new_df = format_shop_category_name(new_df)
    # embeddings = model.encode(new_df["shop_category_name"].to_list())
    # new_df["shop_category_name_cluster"] = kmeans.predict(embeddings) 
    new_df.loc[~new_df["shop_category_name"].isin(top_cats), "shop_category_name"] = "OTHER"
    return new_df

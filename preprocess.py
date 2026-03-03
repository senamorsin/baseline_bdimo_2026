import re
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

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


def process_description(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame(df["description"].apply(parse_description).to_list())], axis=1)



def process_shop_category_name(df: pd.DataFrame, top_cats: list[str], model, kmeans: KMeans):
    new_df = df.copy()
    embeddings = model.encode(new_df["shop_category_name"].to_list(), convert_to_tensor=True)
    new_df["shop_category_name_cluster"] = kmeans.predict(embeddings.cpu().numpy())
    new_df["shop_category_name"] = np.where(new_df["shop_category_name"].isin(top_cats), new_df["shop_category_name"], "OTHER")
    return new_df

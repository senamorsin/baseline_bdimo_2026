from preprocess import format_shop_category_name, process_description, process_vendor_name, process_shop_category_name, extract_description, format_vendor_name, extract_vendor_name, extract_shop_category_name
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_frame(df: pd.DataFrame, top_vendor_names: list[str], top_category_names: list[str]) -> pd.DataFrame:
    df = df.drop("vendor_code", axis=1)
    df = process_description(df)
    df = process_vendor_name(df, top_vendor_names)
    df = process_shop_category_name(df, top_category_names)
    return df

def extract_frame(df: pd.DataFrame, description_extractors: dict, vendor_oh: OneHotEncoder, cats_oh: OneHotEncoder) -> pd.DataFrame:
    df = extract_description(df, description_extractors)
    df = extract_vendor_name(df, vendor_oh)
    df = extract_shop_category_name(df, cats_oh)
    return df
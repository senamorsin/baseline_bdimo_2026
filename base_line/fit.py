import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from preprocess import format_description

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "train.tsv"
MODEL_PATH = ROOT_DIR / "model.joblib"

FEATURE_COLUMNS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]

DEPT_WORD_MAX_FEATURES = 50_000
DEPT_CHAR_MAX_FEATURES = 80_000
DEPT_C = 2.5

CATEGORY_WORD_MAX_FEATURES = 90_000
CATEGORY_CHAR_MAX_FEATURES = 90_000
CATEGORY_KNN_N_NEIGHBORS = 1
CATEGORY_KNN_NGRAM_RANGE = (4, 6)
CATEGORY_SECONDARY_ENABLED = True
CATEGORY_SECONDARY_TEXT_FIELD = "short_text"
CATEGORY_SECONDARY_MAX_FEATURES = 70_000
CATEGORY_SECONDARY_NGRAM_RANGE = (3, 5)
CATEGORY_SECONDARY_TOP_K = 4
CATEGORY_SECONDARY_SIM_SCALE = 1.15
CATEGORY_SECONDARY_SOURCE_BONUS = -0.1
CATEGORY_BLEND_MARGIN = 0.4
CATEGORY_RERANK_TOP_K = 6
CATEGORY_RERANK_WEIGHTS = {
    "sim": 1.41415274143219,
    "dep_match": 0.2853459119796753,
    "vendor_code_match": -0.39958396553993225,
    "vendor_name_match": 0.1455216109752655,
    "shop_match": 0.8270092606544495,
    "title_match": 0.9098311066627502,
    "rank_penalty": 0.1386026293039322,
}
CATEGORY_BLEND_MARGIN_BY_DEPT = {
    2: 0.25,
    4: 0.15,
    6: 0.15,
    7: 0.35,
    10: 0.4,
    11: 0.5,
}
DEPARTMENT_BLEND_MARGIN = 0.25
DEPARTMENT_BLEND_MARGIN_BY_DEPT = {
    0: 0.4,
    1: 0.05,
    3: 0.3,
    4: 0.15,
    5: 0.05,
    7: 0.4,
    8: 0.1,
    9: 0.4,
    10: 0.12,
    11: 0.7,
}

# ---------------------------------------------------------------------------
# COEFFICIENT SEARCH REFERENCE (commented on purpose; no need to rerun)
# ---------------------------------------------------------------------------
# The following snippets are the exact tuning logic used to produce:
# - CATEGORY_BLEND_MARGIN = 0.4
# - CATEGORY_BLEND_MARGIN_BY_DEPT = {2:0.25, 4:0.15, 6:0.15, 7:0.35, 10:0.4, 11:0.5}
# - CATEGORY_WORD_MAX_FEATURES = 90_000
# - CATEGORY_CHAR_MAX_FEATURES = 90_000
# - CATEGORY_RERANK_TOP_K = 6
# - CATEGORY_RERANK_WEIGHTS = {
#       "sim": 1.41415274143219,
#       "dep_match": 0.2853459119796753,
#       "vendor_code_match": -0.39958396553993225,
#       "vendor_name_match": 0.1455216109752655,
#       "shop_match": 0.8270092606544495,
#       "title_match": 0.9098311066627502,
#       "rank_penalty": 0.1386026293039322,
#   }
# - DEPARTMENT_BLEND_MARGIN = 0.25
# - DEPARTMENT_BLEND_MARGIN_BY_DEPT = {
#       0: 0.4, 1: 0.05, 3: 0.3, 4: 0.15, 5: 0.05,
#       7: 0.4, 8: 0.1, 9: 0.4, 10: 0.12, 11: 0.7
#   }
#
# 1) Threshold tuning (coordinate ascent):
# ----------------------------------------------------------------------------
# from sklearn.model_selection import train_test_split
# from scipy import sparse
# import numpy as np
# import fit.fit as F
#
# df = F.read_train_frame("train.tsv")
# tr, va = train_test_split(
#     df, test_size=0.2, random_state=F.RANDOM_STATE, stratify=df["department_id"]
# )
# trp, sw = F.preprocess_frame(tr, fit_description_stopwords=True)
# vap, _ = F.preprocess_frame(va, description_stopwords=sw, fit_description_stopwords=False)
# art = F.build_artifacts(trp)
#
# x_dep = sparse.hstack(
#     [
#         art["dep_word_vectorizer"].transform(vap["text"]),
#         art["dep_char_vectorizer"].transform(vap["short_text"]),
#     ],
#     format="csr",
# )
# dep_pred = art["department_model"].predict(x_dep).astype(np.int64)
# dec = art["department_model"].decision_function(x_dep)
# if dec.ndim == 1:
#     dep_margin = np.abs(dec)
# else:
#     top2 = np.partition(dec, -2, axis=1)[:, -2:]
#     dep_margin = top2[:, 1] - top2[:, 0]
#
# # Then coordinate-ascent over:
# # default threshold in [0.0, 1.0]
# # per-dept overrides in [0.0, 1.0] plus remove-override option
# # keeping the best contest score.
#
# 2) Reranker weight search (random search over top-k candidates):
# ----------------------------------------------------------------------------
# import numpy as np
# from sklearn.model_selection import train_test_split
# from scipy import sparse
# import fit.fit as F
#
# F.CATEGORY_WORD_MAX_FEATURES = 90_000
# F.CATEGORY_CHAR_MAX_FEATURES = 90_000
# F.CATEGORY_KNN_NGRAM_RANGE = (4, 6)
#
# df = F.read_train_frame("train.tsv")
# tr, va = train_test_split(
#     df, test_size=0.2, random_state=F.RANDOM_STATE, stratify=df["department_id"]
# )
# trp, sw = F.preprocess_frame(tr, fit_description_stopwords=True)
# vap, _ = F.preprocess_frame(va, description_stopwords=sw, fit_description_stopwords=False)
# art = F.build_artifacts(trp)
#
# # Build unresolved queries and top-K neighbors from art["category_knn_model"].
# # Candidate features used:
# #   sim, dep_match, vendor_code_match, vendor_name_match, shop_match, title_match, rank
# # Score:
# #   ws*sim + wd*dep_match + wc*vendor_code_match + wv*vendor_name_match
# #   + wshop*shop_match + wt*title_match - wr*rank
# # Select argmax candidate per row, then apply high-confidence department override.
#
# rng = np.random.default_rng(321)
# best = (-1.0, None, None)  # (score, k, weights)
# for k in (8, 10, 12):
#     for _ in range(1500):
#         w = (
#             rng.uniform(0.4, 2.5),   # sim
#             rng.uniform(-0.8, 1.2),  # dep_match
#             rng.uniform(-0.8, 1.2),  # vendor_code_match
#             rng.uniform(-0.8, 1.2),  # vendor_name_match
#             rng.uniform(-0.2, 1.8),  # shop_match
#             rng.uniform(-0.2, 1.8),  # title_match
#             rng.uniform(0.0, 0.5),   # rank_penalty
#         )
#         # evaluate score_metric(...) and update best if improved
#
# # Best for the final setup:
# # k = 6
# # weights = (
# #     1.41415274143219, 0.2853459119796753, -0.39958396553993225,
# #     0.1455216109752655, 0.8270092606544495, 0.9098311066627502,
# #     0.1386026293039322
# # )
#
# 3) Department blend threshold search:
# ----------------------------------------------------------------------------
# # Keep category prediction unchanged, then blend department outputs:
# # dep_out = dep_from_category
# # dep_out[dep_margin >= t] = dep_model_pred[dep_margin >= t]
# # Sweep t in [0.1, 1.0].
# # Best observed on the fixed validation split:
# # t = 0.25
#
# # Then coordinate-ascent over per-department overrides:
# # threshold(dep) in [0.05, 1.2] for departments predicted by dep_model,
# # accepting only changes that improve contest score.
# # Best overrides are DEPARTMENT_BLEND_MARGIN_BY_DEPT above.
# ---------------------------------------------------------------------------
LOOKUP_ORDER = (
    "full",
    "title_vendor_shop",
    "title_shop",
    "title_vendor",
    "vendor_code_unique",
)
DESCRIPTION_STOPWORDS_TOP_K = 100


VENDOR_LIST = {
    "3m",
    "amazon",
    "at electric",
    "babolat",
    "bosch",
    "brita",
    "crossfire",
    "each",
    "espoir",
    "fiskars",
    "hp",
    "jiemiwl",
    "kindle",
    "liitokala",
    "marella",
    "mercury",
    "nauxlu",
    "radiomaster",
    "romiky",
    "sunuo",
    "tbs",
    "vvdi",
    "worx",
    "xhorse",
    "yamaha",
    "jiemi",
    "渲牧",
}


def score_metric(
    y_true_dept: np.ndarray,
    y_pred_dept: np.ndarray,
    y_true_cat: np.ndarray,
    y_pred_cat: np.ndarray,
) -> tuple[float, float, float]:
    dept_f1 = f1_score(y_true_dept, y_pred_dept, average="weighted")
    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    score = 100.0 * (0.3 * dept_f1 + 0.7 * cat_acc)
    return score, dept_f1, cat_acc # type: ignore


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = ""
        prepared[col] = prepared[col].fillna("").astype(str).str.lower().str.strip()
    return prepared


def extract_brand(row: pd.Series) -> str:
    vendor = str(row.get("vendor_name", "") or "").strip().lower()
    if vendor in {"", "нет бренда", "без бренда", "no brand"}:
        text = f"{row.get('title', '')} {row.get('description', '')} {row.get('shop_category_name', '')}".lower()
        best_brand = ""
        best_pos = len(text) + 1
        for brand in VENDOR_LIST:
            match = re.search(rf"\b{re.escape(brand)}\b", text, flags=re.IGNORECASE)
            if match is None:
                continue
            if match.start() < best_pos:
                best_pos = match.start()
                best_brand = brand
        if best_brand:
            return best_brand
    return vendor


def fit_description_stopwords(train_description: pd.Series) -> set[str]:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_description.astype(str))
    idf = pd.DataFrame(
        {"word": vectorizer.get_feature_names_out(), "idf": vectorizer.idf_}
    ).sort_values("idf", ascending=True)
    return set(idf.head(DESCRIPTION_STOPWORDS_TOP_K)["word"].tolist())


def preprocess_frame(
    df: pd.DataFrame,
    description_stopwords: set[str] | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    prepared = normalize_feature_columns(df)
    prepared["vendor_name"] = prepared.apply(extract_brand, axis=1)

    fitted_stopwords = description_stopwords or fit_description_stopwords(prepared["description"])
    prepared = format_description(prepared)
    prepared["description"] = prepared["description"].map(
        lambda x: " ".join(
            token for token in str(x).split() if token not in fitted_stopwords
        )
    )

    prepared["text"] = (
        prepared["title"]
        + " "
        + prepared["description"]
        + " "
        + prepared["shop_category_name"]
        + " "
        + prepared["vendor_name"]
        + " "
        + prepared["vendor_code"]
    )

    prepared["short_text"] = (
        prepared["title"]
        + " "
        + prepared["shop_category_name"]
        + " "
        + prepared["vendor_name"]
    )

    prepared["title_vendor_text"] = (
        prepared["title"]
        + " "
        + prepared["vendor_name"]
        + " "
        + prepared["vendor_code"]
        + " "
        + prepared["shop_category_name"]
    )
    return prepared, fitted_stopwords


def read_train_frame(path: Path) -> pd.DataFrame:
    needed = FEATURE_COLUMNS + ["category_id", "department_id"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in needed)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df["category_id"] = pd.to_numeric(df["category_id"], errors="raise").astype(np.int64)
    df["department_id"] = pd.to_numeric(df["department_id"], errors="raise").astype(np.int64)
    return df


def mode_int(series: pd.Series) -> int:
    return int(series.mode().iat[0]) # type: ignore


def build_map_by_mode(
    df: pd.DataFrame,
    key: pd.Series,
    value_col: str,
    min_count: int = 1,
    min_ratio: float = 0.0,
) -> dict[str, int]:
    grouped = (
        pd.DataFrame({"key": key, "val": df[value_col].to_numpy()})
        .groupby("key")["val"]
        .agg(list)
    )
    result: dict[str, int] = {}
    for k, values in grouped.items():
        if len(values) < min_count:
            continue
        counts = pd.Series(values).value_counts()
        top_count = int(counts.iloc[0])
        if (top_count / len(values)) < min_ratio:
            continue
        result[str(k)] = int(counts.index[0])
    return result


def build_lookup_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    full_key = train_df[FEATURE_COLUMNS].astype(str).agg("||".join, axis=1)
    title_shop_key = train_df["title"] + "||" + train_df["shop_category_name"]
    title_vendor_key = train_df["title"] + "||" + train_df["vendor_name"]
    title_vendor_shop_key = (
        train_df["title"]
        + "||"
        + train_df["vendor_name"]
        + "||"
        + train_df["shop_category_name"]
    )

    full_map = build_map_by_mode(train_df, full_key, "category_id", min_count=1, min_ratio=1.0)
    title_shop_map = build_map_by_mode(
        train_df, title_shop_key, "category_id", min_count=3, min_ratio=0.0
    )
    title_vendor_map = build_map_by_mode(train_df, title_vendor_key, "category_id")
    title_vendor_shop_map = build_map_by_mode(
        train_df, title_vendor_shop_key, "category_id"
    )

    non_empty_vendor = train_df[train_df["vendor_code"] != ""]
    if non_empty_vendor.empty:
        vendor_code_unique_map: dict[str, int] = {}
    else:
        uniq_counts = non_empty_vendor.groupby("vendor_code")["category_id"].nunique()
        unique_codes = uniq_counts[uniq_counts == 1].index
        vendor_code_unique_map = ( # type: ignore
            non_empty_vendor[non_empty_vendor["vendor_code"].isin(unique_codes)]
            .groupby("vendor_code")["category_id"]
            .first()
            .astype(np.int64)
            .to_dict()
        )

    return {
        "full": full_map,
        "title_vendor_shop": title_vendor_shop_map,
        "title_shop": title_shop_map,
        "title_vendor": title_vendor_map,
        "vendor_code_unique": {str(k): int(v) for k, v in vendor_code_unique_map.items()},
    }


def _cast_linear_svc_to_float32(model: LinearSVC) -> None:
    if hasattr(model, "coef_"):
        model.coef_ = model.coef_.astype(np.float32)
    if hasattr(model, "intercept_"):
        model.intercept_ = model.intercept_.astype(np.float32)


def _build_category_query_matrix(
    df: pd.DataFrame,
    row_idx: np.ndarray,
    artifacts: dict,
) -> sparse.csr_matrix:
    word_vectorizer: TfidfVectorizer = artifacts["category_word_vectorizer"]
    char_vectorizer: TfidfVectorizer = artifacts["category_char_vectorizer"]
    return sparse.hstack( # type: ignore
        [
            word_vectorizer.transform(df.loc[row_idx, "text"]),
            char_vectorizer.transform(df.loc[row_idx, "title_vendor_text"]),
        ],
    )


def _rerank_category_neighbors(
    df: pd.DataFrame,
    unresolved_idx: np.ndarray,
    unresolved_query: sparse.csr_matrix,
    dep_margin: np.ndarray,
    dep_for_unresolved: np.ndarray,
    category_knn_model: NearestNeighbors,
    category_knn_labels: np.ndarray,
    category_knn_departments: np.ndarray,
    artifacts: dict,
) -> np.ndarray:
    rerank_cfg = artifacts["category_rerank_config"]

    top_k = int(rerank_cfg.get("top_k", CATEGORY_RERANK_TOP_K))
    top_k = max(1, min(top_k, category_knn_labels.shape[0]))
    distances, neighbor_idx = category_knn_model.kneighbors(
        unresolved_query,
        n_neighbors=top_k,
        return_distance=True,
    )

    secondary_cfg = artifacts["category_secondary_config"]
    secondary_neighbor_idx = None
    secondary_distances = None
    if secondary_cfg is not None:
        secondary_text_field = str(
            secondary_cfg.get("text_field", CATEGORY_SECONDARY_TEXT_FIELD)
        )
        secondary_top_k = int(
            secondary_cfg.get("top_k", CATEGORY_SECONDARY_TOP_K)
        )
        secondary_top_k = max(1, min(secondary_top_k, category_knn_labels.shape[0]))
        secondary_vectorizer: TfidfVectorizer | None = artifacts["category_secondary_vectorizer"]
        secondary_model: NearestNeighbors | None = artifacts["category_secondary_model"]
        assert secondary_vectorizer is not None
        assert secondary_model is not None
        secondary_query = secondary_vectorizer.transform(
            df.loc[unresolved_idx, secondary_text_field]
        )
        secondary_distances, secondary_neighbor_idx = secondary_model.kneighbors(
            secondary_query,
            n_neighbors=secondary_top_k,
            return_distance=True,
        )

    n_rows = neighbor_idx.shape[0]
    n_main = neighbor_idx.shape[1]
    n_secondary = 0 if secondary_neighbor_idx is None else secondary_neighbor_idx.shape[1]
    max_candidates = n_main + n_secondary

    merged_neighbor_idx = np.full((n_rows, max_candidates), -1, dtype=np.int64)
    merged_distances = np.ones((n_rows, max_candidates), dtype=np.float32)
    merged_source = np.full((n_rows, max_candidates), -1, dtype=np.int8)

    for row in range(n_rows):
        pos = 0
        seen: set[int] = set()
        for col in range(n_main):
            idx = int(neighbor_idx[row, col])
            if idx in seen:
                continue
            seen.add(idx)
            merged_neighbor_idx[row, pos] = idx
            merged_distances[row, pos] = distances[row, col]
            merged_source[row, pos] = 0
            pos += 1

        if secondary_neighbor_idx is None:
            continue

        for col in range(n_secondary):
            idx = int(secondary_neighbor_idx[row, col])
            if idx in seen:
                continue
            seen.add(idx)
            merged_neighbor_idx[row, pos] = idx
            merged_distances[row, pos] = secondary_distances[row, col] # type: ignore
            merged_source[row, pos] = 1
            pos += 1

    valid_candidate_mask = merged_neighbor_idx != -1
    safe_neighbor_idx = np.where(valid_candidate_mask, merged_neighbor_idx, 0)

    candidate_labels = np.where(
        valid_candidate_mask,
        category_knn_labels[safe_neighbor_idx],
        -1,
    )
    candidate_departments = np.where(
        valid_candidate_mask,
        category_knn_departments[safe_neighbor_idx],
        -1,
    )
    similarity = (1.0 - merged_distances).astype(np.float32, copy=False)
    if secondary_cfg is not None:
        secondary_sim_scale = float(
            secondary_cfg.get("sim_scale", CATEGORY_SECONDARY_SIM_SCALE)
        )
        similarity = np.where(
            merged_source == 1,
            similarity * secondary_sim_scale,
            similarity,
        )
    similarity = np.where(valid_candidate_mask, similarity, 0.0)

    dep_match = (
        candidate_departments == dep_for_unresolved[:, None]
    ).astype(np.float32)
    dep_match[~valid_candidate_mask] = 0.0

    train_vendor_code = np.asarray(artifacts["category_knn_vendor_code"], dtype=object)
    train_vendor_name = np.asarray(artifacts["category_knn_vendor_name"], dtype=object)
    train_shop = np.asarray(artifacts["category_knn_shop_category_name"], dtype=object)
    train_title = np.asarray(artifacts["category_knn_title"], dtype=object)

    query_vendor_code = df.loc[unresolved_idx, "vendor_code"].to_numpy(dtype=object)
    query_vendor_name = df.loc[unresolved_idx, "vendor_name"].to_numpy(dtype=object)
    query_shop = df.loc[unresolved_idx, "shop_category_name"].to_numpy(dtype=object)
    query_title = df.loc[unresolved_idx, "title"].to_numpy(dtype=object)

    vendor_code_match = (
        train_vendor_code[safe_neighbor_idx] == query_vendor_code[:, None]
    ).astype(np.float32)
    vendor_name_match = (
        train_vendor_name[safe_neighbor_idx] == query_vendor_name[:, None]
    ).astype(np.float32)
    shop_match = (
        train_shop[safe_neighbor_idx] == query_shop[:, None]
    ).astype(np.float32)
    title_match = (
        train_title[safe_neighbor_idx] == query_title[:, None]
    ).astype(np.float32)
    vendor_code_match[~valid_candidate_mask] = 0.0
    vendor_name_match[~valid_candidate_mask] = 0.0
    shop_match[~valid_candidate_mask] = 0.0
    title_match[~valid_candidate_mask] = 0.0
    rank = np.arange(max_candidates, dtype=np.float32)[None, :]

    candidate_score = (
        float(rerank_cfg.get("sim", CATEGORY_RERANK_WEIGHTS["sim"])) * similarity
        + float(rerank_cfg.get("dep_match", CATEGORY_RERANK_WEIGHTS["dep_match"])) * dep_match
        + float(
            rerank_cfg.get(
                "vendor_code_match",
                CATEGORY_RERANK_WEIGHTS["vendor_code_match"],
            )
        )
        * vendor_code_match
        + float(
            rerank_cfg.get(
                "vendor_name_match",
                CATEGORY_RERANK_WEIGHTS["vendor_name_match"],
            )
        )
        * vendor_name_match
        + float(rerank_cfg.get("shop_match", CATEGORY_RERANK_WEIGHTS["shop_match"])) * shop_match
        + float(rerank_cfg.get("title_match", CATEGORY_RERANK_WEIGHTS["title_match"])) * title_match
        - float(
            rerank_cfg.get("rank_penalty", CATEGORY_RERANK_WEIGHTS["rank_penalty"])
        )
        * rank
    )
    if secondary_cfg is not None:
        secondary_source_bonus = float(
            secondary_cfg.get("source_bonus", CATEGORY_SECONDARY_SOURCE_BONUS)
        )
        candidate_score += secondary_source_bonus * (merged_source == 1)
    candidate_score = np.where(valid_candidate_mask, candidate_score, -1e9)

    row_ids = np.arange(len(unresolved_idx))
    selected_pos = np.argmax(candidate_score, axis=1)
    selected_labels = candidate_labels[row_ids, selected_pos]
    selected_departments = candidate_departments[row_ids, selected_pos]

    blend_threshold = np.array(
        [
            CATEGORY_BLEND_MARGIN_BY_DEPT.get(int(dep_id), CATEGORY_BLEND_MARGIN)
            for dep_id in dep_for_unresolved
        ],
        dtype=np.float32,
    )
    high_confidence = dep_margin[unresolved_idx] >= blend_threshold
    dep_candidate_mask = (dep_match > 0) & valid_candidate_mask
    has_dep_candidate = dep_candidate_mask.any(axis=1)
    dep_only_score = np.where(dep_candidate_mask, candidate_score, -1e9)
    dep_best_pos = np.argmax(dep_only_score, axis=1)
    override_mask = (
        high_confidence
        & has_dep_candidate
        & (selected_departments != dep_for_unresolved)
    )
    if override_mask.any():
        override_rows = np.flatnonzero(override_mask)
        selected_labels[override_rows] = candidate_labels[
            override_rows, dep_best_pos[override_rows]
        ]

    return selected_labels.astype(np.int64, copy=False)


def build_artifacts(train_df: pd.DataFrame) -> dict:
    y_department = train_df["department_id"].to_numpy()
    default_category = mode_int(train_df["category_id"])
    default_department = mode_int(train_df["department_id"])
    category_to_department = (
        train_df.groupby("category_id")["department_id"]
        .first()
        .astype(np.int64)
        .to_dict()
    )
    
    dep_word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=DEPT_WORD_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    dep_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=DEPT_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_department: sparse.csr_matrix = sparse.hstack( # type: ignore
        [
            dep_word_vectorizer.fit_transform(train_df["text"]),
            dep_char_vectorizer.fit_transform(train_df["short_text"]),
        ],
    )
    department_model = LinearSVC(C=DEPT_C, random_state=RANDOM_STATE)
    department_model.fit(x_department, y_department)
    _cast_linear_svc_to_float32(department_model)

    category_word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=CATEGORY_WORD_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    category_char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CATEGORY_KNN_NGRAM_RANGE,
        min_df=1,
        max_features=CATEGORY_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_category_knn: sparse.csr_matrix = sparse.hstack( # type: ignore
        [
            category_word_vectorizer.fit_transform(train_df["text"]),
            category_char_vectorizer.fit_transform(train_df["title_vendor_text"]),
        ],
    )
    category_knn_model = NearestNeighbors(
        n_neighbors=CATEGORY_KNN_N_NEIGHBORS,
        metric="cosine",
        algorithm="brute",
    )
    category_knn_model.fit(x_category_knn)

    secondary_vectorizer = None
    secondary_model = None
    secondary_cfg = None
    if CATEGORY_SECONDARY_ENABLED:
        secondary_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=CATEGORY_SECONDARY_NGRAM_RANGE,
            min_df=1,
            max_features=CATEGORY_SECONDARY_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_secondary = secondary_vectorizer.fit_transform(
            train_df[CATEGORY_SECONDARY_TEXT_FIELD]
        )
        secondary_model = NearestNeighbors(
            n_neighbors=1,
            metric="cosine",
            algorithm="brute",
        )
        secondary_model.fit(x_secondary)
        secondary_cfg = {
            "text_field": CATEGORY_SECONDARY_TEXT_FIELD,
            "top_k": CATEGORY_SECONDARY_TOP_K,
            "sim_scale": CATEGORY_SECONDARY_SIM_SCALE,
            "source_bonus": CATEGORY_SECONDARY_SOURCE_BONUS,
        }

    return {
        "dep_word_vectorizer": dep_word_vectorizer,
        "dep_char_vectorizer": dep_char_vectorizer,
        "department_model": department_model,
        "category_word_vectorizer": category_word_vectorizer,
        "category_char_vectorizer": category_char_vectorizer,
        "category_knn_model": category_knn_model,
        "category_knn_labels": train_df["category_id"].to_numpy(dtype=np.int64),
        "category_knn_departments": train_df["department_id"].to_numpy(dtype=np.int64),
        "category_knn_vendor_code": train_df["vendor_code"].to_numpy(dtype=object),
        "category_knn_vendor_name": train_df["vendor_name"].to_numpy(dtype=object),
        "category_knn_shop_category_name": train_df["shop_category_name"].to_numpy(
            dtype=object
        ),
        "category_knn_title": train_df["title"].to_numpy(dtype=object),
        "category_rerank_config": {
            "top_k": CATEGORY_RERANK_TOP_K,
            **CATEGORY_RERANK_WEIGHTS,
        },
        "category_secondary_vectorizer": secondary_vectorizer,
        "category_secondary_model": secondary_model,
        "category_secondary_config": secondary_cfg,
        "lookup_maps": build_lookup_maps(train_df),
        "lookup_order": LOOKUP_ORDER,
        "default_category": int(default_category),
        "default_department": int(default_department),
        "category_to_department": {
            int(cat): int(dep) for cat, dep in category_to_department.items() # type: ignore
        },
    }


def _apply_lookup_maps(df: pd.DataFrame, artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    category_pred = np.full(len(df), -1, dtype=np.int64)
    unresolved = np.ones(len(df), dtype=bool)

    lookup_maps: dict[str, dict[str, int]] = artifacts["lookup_maps"]
    lookup_order: tuple[str, ...] = tuple(artifacts["lookup_order"])

    for key in lookup_order:
        mapping = lookup_maps.get(key, {})
        if not mapping:
            continue

        if key == "full":
            series = df[FEATURE_COLUMNS].astype(str).agg("||".join, axis=1)
        elif key == "title_vendor_shop":
            series = df["title"] + "||" + df["vendor_name"] + "||" + df["shop_category_name"]
        elif key == "title_shop":
            series = df["title"] + "||" + df["shop_category_name"]
        elif key == "title_vendor":
            series = df["title"] + "||" + df["vendor_name"]
        elif key == "title":
            series = df["title"]
        elif key == "vendor_code_unique":
            series = df["vendor_code"]
        elif key == "vendor_shop":
            series = df["vendor_name"] + "||" + df["shop_category_name"]
        elif key == "shop":
            series = df["shop_category_name"]
        else:
            continue

        mapped = series.map(mapping)
        match_mask = unresolved & mapped.notna().to_numpy()
        if not match_mask.any():
            continue

        category_pred[match_mask] = mapped[match_mask].astype(np.int64).to_numpy()
        unresolved[match_mask] = False

    return category_pred, unresolved


def predict_with_artifacts(df: pd.DataFrame, artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    category_pred, unresolved = _apply_lookup_maps(df, artifacts)
    department_model: LinearSVC = artifacts["department_model"]
    dep_word_vectorizer: TfidfVectorizer = artifacts["dep_word_vectorizer"]
    dep_char_vectorizer: TfidfVectorizer = artifacts["dep_char_vectorizer"]
    category_knn_model: NearestNeighbors = artifacts["category_knn_model"]
    category_knn_labels: np.ndarray = artifacts["category_knn_labels"]
    category_knn_departments: np.ndarray = artifacts["category_knn_departments"]
    default_category = int(artifacts["default_category"])
    default_department = int(artifacts["default_department"])
    category_to_department: dict[int, int] = {
        int(cat): int(dep)
        for cat, dep in artifacts["category_to_department"].items()
    }

    x_department: sparse.csr_matrix = sparse.hstack( # type: ignore
        [
            dep_word_vectorizer.transform(df["text"]),
            dep_char_vectorizer.transform(df["short_text"]),
        ],
    )
    department_pred = department_model.predict(x_department).astype(np.int64)
    dep_decision = department_model.decision_function(x_department)
    if dep_decision.ndim == 1:
        dep_margin = np.abs(dep_decision)
    else:
        top2 = np.partition(dep_decision, -2, axis=1)[:, -2:]
        dep_margin = top2[:, 1] - top2[:, 0]

    if unresolved.any():
        unresolved_idx = np.flatnonzero(unresolved)
        unresolved_query = _build_category_query_matrix(df, unresolved_idx, artifacts)
        dep_for_unresolved = department_pred[unresolved_idx]
        unresolved_cat_pred = _rerank_category_neighbors(
            df=df,
            unresolved_idx=unresolved_idx,
            unresolved_query=unresolved_query,
            dep_margin=dep_margin,
            dep_for_unresolved=dep_for_unresolved,
            category_knn_model=category_knn_model,
            category_knn_labels=category_knn_labels,
            category_knn_departments=category_knn_departments,
            artifacts=artifacts,
        )
        category_pred[unresolved_idx] = unresolved_cat_pred

    unresolved_after = category_pred == -1
    if unresolved_after.any():
        category_pred[unresolved_after] = default_category

    dep_from_category = np.array(
        [
            category_to_department.get(int(cat), default_department)
            for cat in category_pred
        ],
        dtype=np.int64,
    )
    blend_threshold = np.array(
        [
            DEPARTMENT_BLEND_MARGIN_BY_DEPT.get(int(dep_id), DEPARTMENT_BLEND_MARGIN)
            for dep_id in department_pred
        ],
        dtype=np.float32,
    )
    use_department_model = dep_margin >= blend_threshold
    dep_from_category[use_department_model] = department_pred[use_department_model]

    return category_pred, dep_from_category


def evaluate_once(df: pd.DataFrame) -> None:
    print("Running validation split...", flush=True)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["department_id"],
    )

    train_prepared, stopwords = preprocess_frame(
        train_df
    )
    val_prepared, _ = preprocess_frame(
        val_df, description_stopwords=stopwords
    )

    artifacts = build_artifacts(train_prepared)
    cat_pred, dep_pred = predict_with_artifacts(val_prepared, artifacts)

    score, dept_f1, cat_acc = score_metric(
        val_prepared["department_id"].to_numpy(),
        dep_pred,
        val_prepared["category_id"].to_numpy(),
        cat_pred,
    )
    print(f"Validation department weighted F1: {dept_f1:.4f}", flush=True)
    print(f"Validation category accuracy:      {cat_acc:.4f}", flush=True)
    print(f"Validation contest score:          {score:.2f}", flush=True)


def train_and_save(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> None:
    print("Fitting final models on full train.tsv...", flush=True)
    prepared, stopwords = preprocess_frame(df)
    artifacts = build_artifacts(prepared)
    artifacts["description_stopwords"] = sorted(stopwords)
    joblib.dump(artifacts, model_path, compress=3)
    print(f"Saved model to {model_path}", flush=True)


def main():
    df = read_train_frame(TRAIN_PATH)
    evaluate_once(df)
    train_and_save(df)


if __name__ == "__main__":
    main()

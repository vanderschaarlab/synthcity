# stdlib
from typing import Dict, List

# third party
import pandas as pd
from pydantic import validate_arguments
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity relative
from .evaluation import evaluate_classifier, evaluate_regression


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def compress_dataset(
    df: pd.DataFrame,
    cat_limit: int = 10,
    impute: bool = True,
    score_threshold: float = 0.98,
) -> pd.DataFrame:
    df = df.copy()
    original_dtypes = df.infer_objects().dtypes

    if impute:
        df = df.fillna(0)  # TODO: should we use a special symbol?
    df.columns = df.columns.astype(str)

    # check redundant columns
    covariates = df.columns
    redundant: List[str] = []

    # encode
    encoders = {}
    for col in df.columns:
        if df[col].dtype not in ["object", "category"]:
            continue

        encoders[col] = LabelEncoder().fit(df[col])
        df[col] = encoders[col].transform(df[col])

    # compress
    compressers = {}
    for column in covariates:
        X = df[covariates].drop(columns=redundant + [column])
        y = df[column]

        if len(df[column].unique()) < cat_limit:
            model = XGBClassifier()
            try:
                score = evaluate_classifier(model, X, y)["clf"]["aucroc"][0]
            except BaseException:
                continue

        else:
            model = XGBRegressor()

            try:
                score = evaluate_regression(model, X, y)["clf"]["r2"][0]
            except BaseException:
                continue

        if score >= score_threshold:
            redundant.append(column)
            model.fit(X, y)

            src_cols = X.columns
            compressers[column] = {
                "cols": list(src_cols),
                "model": model,
                "min": y.min(),
                "max": y.max(),
            }
    df = df.drop(columns=redundant)
    covariates = df.columns

    # compress categoricals
    compressers_categoricals = {}
    categoricals: List[List[str]] = [[]]

    for column in covariates:
        if len(df[column].unique()) > cat_limit:
            continue

        categoricals[-1].append(column)

        if len(df[categoricals[-1]].drop_duplicates()) >= cat_limit:
            categoricals.append([])

    for cats in categoricals:
        if len(cats) <= 1:
            continue
        cat_types = df[cats].infer_objects().dtypes

        aggr = df[cats].astype(str).agg(" ".join, 1)

        encoder = LabelEncoder().fit(aggr)
        encoded = encoder.transform(aggr)
        encoded_col = " ".join(cats)
        df[encoded_col] = encoded
        df = df.drop(columns=cats)

        compressers_categoricals[encoded_col] = {
            "cols": cats,
            "model": encoder,
            "types": cat_types,
        }

    context = {
        "encoders": encoders,
        "compressers": compressers,
        "compressers_categoricals": compressers_categoricals,
        "original_dtypes": original_dtypes,
    }
    return df, context


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def decompress_dataset(
    df: pd.DataFrame, context: Dict, cat_limit: int = 10
) -> pd.DataFrame:
    if "encoders" not in context:
        raise ValueError("Invalid context. missing encoders")

    if "compressers" not in context:
        raise ValueError("Invalid context. missing compressers")

    if "compressers_categoricals" not in context:
        raise ValueError("Invalid context. missing compressers_categoricals")

    df = df.copy()
    df.columns = df.columns.astype(str)

    # decompress categoricals
    for cat_group in context["compressers_categoricals"]:
        if cat_group not in df.columns:
            raise ValueError(f"Invalid categories {cat_group}")

        encoder = context["compressers_categoricals"][cat_group]["model"]
        src_cols = context["compressers_categoricals"][cat_group]["cols"]
        dtypes = context["compressers_categoricals"][cat_group]["types"]

        df[cat_group] = encoder.inverse_transform(df[cat_group])
        decoded = df[cat_group].str.split(" ", n=-1, expand=True)

        if decoded.shape[1] != len(src_cols):
            raise ValueError(
                f"Invalid decoding shape {decoded.shape} expected {len(src_cols)}"
            )

        df[src_cols] = decoded.astype(dtypes.reset_index(drop=True))
        df = df.drop(columns=[cat_group])

    # decompress redundant

    for i in range(len(context["compressers"].keys())):
        todo_cols = list(context["compressers"].keys())
        if pd.Series(todo_cols).isin(df.columns).sum() == len(todo_cols):
            break

        for col in context["compressers"]:
            if col in df.columns:
                continue

            model = context["compressers"][col]["model"]
            src_cols = context["compressers"][col]["cols"]
            vmin = context["compressers"][col]["min"]
            vmax = context["compressers"][col]["max"]

            if pd.Series(src_cols).isin(df.columns).sum() != len(
                src_cols
            ):  # need to decode something else first
                continue

            src_covs = df[src_cols]
            df[col] = model.predict(src_covs)
            df[col].clip(lower=vmin, upper=vmax, inplace=True)

    # decode categoricals
    for col in context["encoders"]:
        if col not in df:
            raise ValueError(f"Missing {col} for decoding")
        df[col] = context["encoders"][col].inverse_transform(df[col])

    original_dtypes = context["original_dtypes"]
    df = df.astype(original_dtypes.filter(df.columns))

    return df

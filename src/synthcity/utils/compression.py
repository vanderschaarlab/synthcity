# stdlib
from typing import List

# third party
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity relative
from .evaluation import evaluate_classifier, evaluate_regression


def compress_dataset(df: pd.DataFrame, cat_limit: int = 15) -> pd.DataFrame:
    # check redundant columns
    covariates = df.columns
    redundant: List[str] = []

    for column in covariates:
        X = df[covariates].drop(columns=redundant + [column])
        y = df[column]

        if len(df[column].unique()) < 10:
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

        if score > 0.95:
            redundant.append(column)

    df = df.drop(columns=redundant)
    covariates = df.columns

    # compress categoricals
    categoricals: List[List[str]] = [[]]

    for column in covariates:
        if len(df[column].unique()) > cat_limit:
            continue

        categoricals[-1].append(column)

        if len(df[categoricals[-1]].drop_duplicates()) >= cat_limit:
            categoricals.append([])

    for cats in categoricals:
        if len(cats) == 1:
            continue
        aggr = df[cats].astype(str).agg(" ".join, 1)

        encoded = LabelEncoder().fit_transform(aggr)
        df[" ".join(cats)] = encoded
        df = df.drop(columns=cats)

    return df

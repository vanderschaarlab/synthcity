# stdlib
import math
from typing import List, Tuple, Union, Literal, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
) -> dict:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res


def get_features(X: pd.DataFrame, sensitive_features: List[str] = []) -> List:
    """Return the non-sensitive features from dataset X"""
    features = list(X.columns)
    for col in sensitive_features:
        if col in features:
            features.remove(col)

    return features


def get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
    if nclasses == 2:
        if len(y_pred_proba.shape) < 2:
            return y_pred_proba

        if y_pred_proba.shape[1] == 2:
            return y_pred_proba[:, 1]

    return y_pred_proba


def evaluate_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: Union[np.ndarray, None] = None,
) -> Tuple[float, float]:

    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = len(set(np.ravel(y_test)))

    y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    if n_classes > 2:

        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_auc: dict = dict()

        if classes is None:
            classes = sorted(set(np.ravel(y_test)))

        y_test = label_binarize(y_test, classes=classes)

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_test, y_pred_proba_tmp, average="micro"
        )

        aucroc = roc_auc["micro"]
        aucprc = average_precision["micro"]
    else:

        aucroc = roc_auc_score(np.ravel(y_test), y_pred_proba_tmp, multi_class="ovr")
        aucprc = average_precision_score(np.ravel(y_test), y_pred_proba_tmp)

    return aucroc, aucprc


def calculate_fair_aug_sample_size(
    X_train: pd.DataFrame,
    fairness_column: str,  # a categorical column of K levels
    rule: Literal[
        "equal", "log", "ad-hoc"
    ],  # TODO: Confirm are there any more methods to include
    ad_hoc_augment_vals: Dict[
        Union[int, str], int
    ] = {},  # Only required for rule == "ad-hoc"
) -> Dict:
    """Calculate how many samples to augment.

    Args:
        X_train (pd.DataFrame): The real dataset to be augmented.
        fairness_column (str): The column name of the column to test the fairness of a downstream model with respect to.
        ad_hoc_augment_vals (Dict[ Union[int, str], int ], optional): A dictionary containing the number of each class to augment the real data with. If using rule="ad-hoc" this function returns ad_hoc_augment_vals, otherwise this parameter is ignored. Defaults to {}.

    Returns:
        Dict: A dictionary containing the number of each class to augment the real data with.
    """

    # the majority class is unchanged

    if rule == "equal":
        # number of sample will be the same for each value in the fairness column after augmentation
        # N_aug(i) = N_ang(j) for all i and j in value in the fairness column
        fairness_col_counts = X_train[fairness_column].value_counts()
        majority_size = fairness_col_counts.max()
        augmentation_counts = {
            fair_col_val: (majority_size - fairness_col_counts.loc[fair_col_val])
            for fair_col_val in fairness_col_counts.index
        }
    elif rule == "log":
        # number of samples in aug data will be proportional to the log frequency in the real data.
        # Note: taking the log makes the distribution more even.
        # N_aug(i) is proportional to log(N_real(i))
        fairness_col_counts = X_train[fairness_column].value_counts()
        majority_size = fairness_col_counts.max()
        log_coefficient = majority_size / math.log(majority_size)

        augmentation_counts = {
            fair_col_val: (
                majority_size - round(math.log(fair_col_count) * log_coefficient)
            )
            for fair_col_val, fair_col_count in fairness_col_counts.items()
        }
    elif rule == "ad-hoc":
        # use user-specified values to augment
        assert set(ad_hoc_augment_vals.keys()) == set(X_train[fairness_column].values)
        augmentation_counts = ad_hoc_augment_vals

    # return dictionary of how much to augment for each value in the fairness column
    return augmentation_counts

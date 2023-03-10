# stdlib
import math
from copy import copy
from typing import Any, Dict, Literal, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader


def calculate_fair_aug_sample_size(
    X_train: pd.DataFrame,
    fairness_column: Optional[str],  # a categorical column of K levels
    rule: Literal[
        "equal", "log", "ad-hoc"
    ],  # TODO: Confirm are there any more methods to include
    ad_hoc_augment_vals: Optional[
        Dict[Any, int]
    ] = None,  # Only required for rule == "ad-hoc"
) -> Dict:
    """Calculate how many samples to augment.

    Args:
        X_train (pd.DataFrame): The real dataset to be augmented.
        fairness_column (str): The column name of the column to test the fairness of a downstream model with respect to.
        ad_hoc_augment_vals (Dict[ Union[int, str], int ], optional): A dictionary containing the number of each class to augment the real data with. If using rule="ad-hoc" this function returns ad_hoc_augment_vals, otherwise this parameter is ignored. Defaults to {}.

    Returns:
        Dict: A dictionary containing the number of each class to augment the real data with.
    """
    if X_train.get_fairness_column() is None:
        raise TypeError(
            "To use the augmentation metrics, `fairness_column` must be set to a string representing the name of a column in the DataLoader."
        )

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
        if not ad_hoc_augment_vals:
            raise AssertionError(
                "When augmenting with an `ad-hoc` method, ad_hoc_augment_vals must be a dictionary, where the dictionary keys are the values of the fairness_column and the dictionary values are the number of records to augment."
            )
        else:
            if set(ad_hoc_augment_vals.keys()) != set(X_train[fairness_column].values):
                raise AssertionError(
                    "ad_hoc_augment_vals must be a dictionary, where the dictionary keys are the values of the fairness_column and the dictionary values are the number of records to augment."
                )
            augmentation_counts = ad_hoc_augment_vals

    return augmentation_counts


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _generate_synthetic_data(
    X_train: DataLoader,
    augment_generator: Any,
    strict: bool = True,
    rule: Literal["equal", "log", "ad-hoc"] = "equal",
    ad_hoc_augment_vals: Optional[
        Dict[Any, int]
    ] = None,  # Only required for rule == "ad-hoc"
    synthetic_constraints: Optional[Constraints] = None,
    **generate_kwargs: Any,
) -> pd.DataFrame:

    """Generates synthetic data

    Args:
        X_train (pd.DataFrame): The dataset used to train the downstream model.
        y_train (Union[pd.Series, pd.DataFrame]): The data labels for `X_train`. This is used to train the downstream model.
        fairness_column (str): The column name of the column to test the fairness of a downstream model with respect to.
        target_column (str): The column name of the label column.
        syn_model_name (str): The name of the synthetic model plugin to use to generate the synthetic data.
        strict (bool, optional): Flag to ensure that the condition for generating synthetic data is strictly met. Defaults to False.
        rule (Literal["equal", "log", "ad-hoc"): The rule used to achieve the desired proportion records with each value in the fairness column. Defaults to "equal".
        ad_hoc_augment_vals (Dict[ Union[int, str], int ], optional): A dictionary containing the number of each class to augment the real data with. This is only required if using the rule="ad-hoc" option. Defaults to {}.
        random_state (int, optional): The random state to seed the synthetic data generation. Defaults to 42.

    Returns:
        pd.DataFrame: The generated synthetic data.
    """
    augmentation_counts = calculate_fair_aug_sample_size(
        X_train.dataframe(),
        X_train.get_fairness_column(),
        rule,
        ad_hoc_augment_vals=ad_hoc_augment_vals,
    )
    if not strict:
        # set count equal to the total number of records required according to calculate_fair_aug_sample_size
        count = sum(augmentation_counts.values())
        cond = pd.Series(
            np.repeat(
                list(augmentation_counts.keys()), list(augmentation_counts.values())
            )
        )
        syn_data = augment_generator.generate(
            count=count,
            cond=cond,
            constraints=synthetic_constraints,
            **generate_kwargs,
        ).dataframe()
    else:
        syn_data_list = []
        for fairness_value, count in augmentation_counts.items():
            if count > 0:
                constraints = Constraints(
                    rules=[(X_train.get_fairness_column(), "==", fairness_value)]
                )
                syn_data_list.append(
                    augment_generator.generate(
                        count=count, constraints=constraints
                    ).dataframe()
                )
        syn_data = pd.concat(syn_data_list)
    return syn_data


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def augment_data(
    X_train: DataLoader,
    augment_generator: Any,
    strict: bool = False,
    rule: Literal["equal", "log", "ad-hoc"] = "equal",
    ad_hoc_augment_vals: Optional[
        Dict[Any, int]
    ] = None,  # Only required for rule == "ad-hoc"
    synthetic_constraints: Optional[Constraints] = None,
    **generate_kwargs: Any,
) -> DataLoader:
    """Augment the real data with generated synthetic data

    Args:
        X (DataLoader): The ground truth DataLoader to augment with synthetic data.
        model_name (str): The name of the synthetic model plugin to use to generate the synthetic data.
        prefix (str, optional): prefix (str): The prefix for the saved synthetic data generation model filename. Defaults to "fairness.conditional_augmentation".
        strict (bool, optional): Flag to ensure that the condition for generating synthetic data is strictly met. Defaults to False.
        rule (Literal["equal", "log", "ad-hoc"): The rule used to achieve the desired proportion records with each value in the fairness column. Defaults to "equal".
        ad_hoc_augment_vals (Dict[ Union[int, str], int ], optional): A dictionary containing the number of each class to augment the real data with. This is only required if using the rule="ad-hoc" option. Defaults to {}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The augmented dataset and labels.
    """
    syn_data = _generate_synthetic_data(
        X_train,
        augment_generator,
        strict=strict,
        rule=rule,
        ad_hoc_augment_vals=ad_hoc_augment_vals,
        synthetic_constraints=synthetic_constraints,
        **generate_kwargs,
    )

    augmented_data_loader = copy(X_train)
    augmented_data_loader.data = pd.concat(
        [
            X_train.data,
            syn_data,
        ]
    )

    return augmented_data_loader

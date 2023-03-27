# mypy: ignore-errors

# stdlib
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

# third party
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from sklearn.impute import SimpleImputer

# synthcity absolute
from synthcity.utils.dataframe import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    task_type: TaskType
    n_classes: Optional[int]

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINARY

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        assert policy is None
        return dataset

    assert policy is not None
    if policy == "drop-rows":
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            "test"
        ].all(), "Cannot drop test rows, since this will affect the final metrics."
        new_data = {}
        for data_name in ["X_num", "X_cat", "y"]:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == "mean":
        new_values = np.nanmean(dataset.X_num["train"], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert raise_unknown("policy", policy)
    return dataset


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict,
    normalization: Normalization,
    seed: Optional[int],
    return_normalizer: bool = False,
) -> ArrayDict:
    X_train = X["train"]
    if normalization == "standard":
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
    else:
        raise_unknown("normalization", normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):
        if policy is None:
            X_new = X
        elif policy == "most_frequent":
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
            imputer.fit(X["train"])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            raise_unknown("categorical NaN policy", policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X["train"]) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X["train"].shape[1]):
        counter = Counter(X["train"][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def build_target(y: ArrayDict, task_type: TaskType) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    if task_type == TaskType.REGRESSION:
        mean, std = float(y["train"].mean()), float(y["train"].std())
        y = {k: (v - mean) / std for k, v in y.items()}
        info["mean"] = mean
        info["std"] = std
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True,
        )
        num_transform = num_transform

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)

    y, y_info = build_target(dataset.y, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    return dataset


def make_dataset(
    df: pd.DataFrame,
    target: str,
    cat_counts: Dict[str, int],
    T: Transformations,
) -> Dataset:
    # classification
    if len(cat_counts) > 0:
        task_type = TaskType.CLASSIFICATION
    else:
        task_type = TaskType.REGRESSION

    X_cat = df[list(cat_counts.keys())]
    X_num = df.drop(columns=list(X_cat.keys()) + [target])
    y = df[target]

    D = Dataset(
        X_num,
        X_cat,
        y,
        task_type=TaskType(task_type),
    )

    return transform_dataset(D, T, None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != "cpu":
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


##############
# DataLoader #
##############


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, split: Literal["train", "val", "test"]):
        super().__init__()

        self.X_num = (
            torch.from_numpy(dataset.X_num[split])
            if dataset.X_num is not None
            else None
        )
        self.X_cat = (
            torch.from_numpy(dataset.X_cat[split])
            if dataset.X_cat is not None
            else None
        )
        self.y = torch.from_numpy(dataset.y[split])

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            "y": self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict


# def prepare_dataloader(
#     dataset: Dataset,
#     split: str,
#     batch_size: int,
# ):
#     torch_dataset = TabDataset(dataset, split)
#     loader = torch.utils.data.DataLoader(
#         torch_dataset,
#         batch_size=batch_size,
#         shuffle=(split == "train"),
#         num_workers=1,
#     )
#     while True:
#         yield from loader


# def prepare_torch_dataloader(
#     dataset: Dataset,
#     split: str,
#     shuffle: bool,
#     batch_size: int,
# ) -> torch.utils.data.DataLoader:

#     torch_dataset = TabDataset(dataset, split)
#     loader = torch.utils.data.DataLoader(
#         torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
#     )

#     return loader


def concat_features(D: Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {
            k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()
        }
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {
            k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()
        }
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X


def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat(
            [
                pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
                pd.DataFrame(y, columns=["y"]),
            ],
            axis=1,
        )
    if X_cat is not None:
        return pd.concat(
            [
                pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
                pd.DataFrame(
                    X_cat,
                    columns=list(
                        range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1])
                    ),
                ),
                pd.DataFrame(y, columns=["y"]),
            ],
            axis=1,
        )
    return pd.concat(
        [
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=["y"]),
        ],
        axis=1,
    )


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")

# stdlib
from typing import Any, Generator, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split

# synthcity absolute
from synthcity.plugins.core.models.tabular_encoder import FeatureInfo
from synthcity.utils.constants import DEVICE


class BaseSampler(torch.utils.data.sampler.Sampler):
    """DataSampler samples the conditional vector and corresponding data."""

    def get_dataset_conditionals(self) -> np.ndarray:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(self, batch: int) -> Optional[Tuple]:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional_for_class(self, batch: int, c: int) -> Optional[np.ndarray]:
        return None

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return 0

    def conditional_probs(self) -> Optional[np.ndarray]:
        """Return the total number of categories."""
        return None

    def train_test(self) -> Tuple:
        raise NotImplementedError()


class ImbalancedDatasetSampler(BaseSampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, labels: List, train_size: float = 0.8) -> None:
        # if indices is not provided, all elements in the dataset will be considered
        indices = list(range(len(labels)))
        self.train_idx, self.test_idx = train_test_split(indices, train_size=train_size)
        self.train_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(self.train_idx)
        }

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_train_samples = len(self.train_idx)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = indices

        df = df.loc[self.train_idx]

        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Generator:
        return (
            self.train_mapping[self.train_idx[i]]
            for i in torch.multinomial(
                self.weights, self.num_train_samples, replacement=True
            )
        )

    def __len__(self) -> int:
        return len(self.train_idx)

    def train_test(self) -> Tuple:
        return self.train_idx, self.test_idx


class ConditionalDatasetSampler(BaseSampler):
    """DataSampler samples the conditional vector and corresponding data."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        output_info: List[FeatureInfo],
        device: Any = DEVICE,
        train_size: float = 0.8,
    ) -> None:
        self._device = device

        indices = np.arange(0, len(data))
        self._train_idx, self._test_idx = train_test_split(
            indices, train_size=train_size
        )
        self._train_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(self._train_idx)
        }
        self._num_items = len(indices)

        self._internal_setup(data, output_info)

        self._prepare_dataset_conditionals()

    def _random_choice_prob_index(self, discrete_column_id: int) -> np.ndarray:
        probs = self._discrete_feat_value_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def get_dataset_conditionals(self) -> np.ndarray:
        return self._dataset_conditional

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(
        self, batch: int, with_ids: bool = False, p: Optional[np.ndarray] = None
    ) -> Any:
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
            p: Optional np.ndarray
                Optional probability for each discrete column
        """
        if self._n_discrete_columns == 0:
            return None

        if p is not None:
            if p.shape[-1] != self._n_conditional_dimension:
                raise ValueError(f"Invalid probability shape {p.shape}")

            cond_res = np.zeros((batch, self._n_conditional_dimension), dtype="float32")

            ind = np.random.choice(self._n_conditional_dimension, batch, p=p)

            cond_res[np.arange(batch), ind] = 1

            return cond_res

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch
        )

        cond = np.zeros((batch, self._n_conditional_dimension)).astype(float)
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (
            self._categorical_feat_offset[discrete_column_id] + category_id_in_col
        )
        cond[np.arange(batch), category_id] = 1

        if with_ids:
            return cond, discrete_column_id, category_id_in_col

        return cond

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional_for_class(self, batch: int, c: int) -> Optional[np.ndarray]:
        cond = np.zeros((batch, self._n_conditional_dimension)).astype(float)
        cond[..., c] = 1

        return cond

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional_indices(
        self,
        cat_feats: np.ndarray,
        cat_values: np.ndarray,
    ) -> List[int]:
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if len(cat_values) != len(cat_feats):
            raise ValueError(f"Invalid categorical features {cat_values}")

        idx = []
        for c, o in zip(cat_feats, cat_values):
            idx.append(np.random.choice(self._categorical_value_to_row_[c][o]))

        return idx

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return self._n_conditional_dimension

    def conditional_probs(self) -> Optional[np.ndarray]:
        return self._conditional_probs

    def __iter__(self) -> Generator:
        np.random.shuffle(self._train_idx)

        for idx in self._train_idx:
            yield self._train_mapping[idx]

    def __len__(self) -> int:
        return len(self._train_idx)

    def _internal_setup(
        self,
        data: pd.DataFrame,
        output_info: List[FeatureInfo],
    ) -> None:
        if data.shape[1] != sum([item.output_dimensions for item in output_info]):
            raise ValueError("Invalid data shape {data.shape}")

        def is_discrete_column(column_info: FeatureInfo) -> bool:
            return column_info.feature_type == "discrete"

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)]
        )

        data = np.asarray(data)
        # Store the row id for each category in each categorical column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._categorical_value_to_row_ = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                rid_by_cat = []
                for j in range(column_info.output_dimensions):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._categorical_value_to_row_.append(rid_by_cat)

            st += column_info.output_dimensions

        if st != data.shape[1]:
            raise RuntimeError(f"Invalid offset {st} {data.shape}")

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [
                column_info.output_dimensions
                for column_info in output_info
                if is_discrete_column(column_info)
            ],
            default=0,
        )

        self._categorical_feat_offset = np.zeros(n_discrete_columns, dtype="int32")
        self._categorical_feat_dimension = np.zeros(n_discrete_columns, dtype="int32")
        self._discrete_feat_value_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_conditional_dimension = sum(
            [
                column_info.output_dimensions
                for column_info in output_info
                if is_discrete_column(column_info)
            ]
        )
        self._conditional_probs = np.zeros(self._n_conditional_dimension)

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                ed = st + column_info.output_dimensions
                cond_ed = current_cond_st + column_info.output_dimensions

                category_freq = np.sum(data[:, st:ed], axis=0)
                self._conditional_probs[current_cond_st:cond_ed] = category_freq
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_feat_value_prob[
                    current_id, : column_info.output_dimensions
                ] = category_prob
                self._categorical_feat_offset[current_id] = current_cond_st
                self._categorical_feat_dimension[
                    current_id
                ] = column_info.output_dimensions

                current_cond_st += column_info.output_dimensions
                current_id += 1

            st += column_info.output_dimensions
        if st != data.shape[1]:
            raise ValueError(f"Invalid offset {st} {data.shape}")

        self._conditional_probs = self._conditional_probs / (
            np.sum(self._conditional_probs) + 1e-8
        )

    def _prepare_dataset_conditionals(self) -> None:
        self._dataset_conditional = None
        if self._n_discrete_columns == 0:
            return

        (
            self._dataset_conditional,
            categoricals,
            categoricals_vals,
        ) = self.sample_conditional(self._num_items, with_ids=True)

        sampling_indices = self.sample_conditional_indices(
            categoricals, categoricals_vals
        )

        self._train_idx = [idx for idx in sampling_indices if idx in self._train_idx]
        self._train_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(self._train_idx)
        }

    def train_test(self) -> Tuple:
        return self._train_idx, self._test_idx

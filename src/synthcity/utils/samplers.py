# stdlib
from typing import Any, Generator, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.models.tabular_encoder import ColumnTransformInfo
from synthcity.utils.constants import DEVICE


class BaseSampler(torch.utils.data.sampler.Sampler):
    """DataSampler samples the conditional vector and corresponding data."""

    def get_train_conditionals(self) -> np.ndarray:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(self, batch: int) -> Optional[Tuple]:
        return None

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return 0


class ImbalancedDatasetSampler(BaseSampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, labels: List) -> None:
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(labels)))

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Generator:
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self) -> int:
        return self.num_samples


class ConditionalDatasetSampler(BaseSampler):
    """DataSampler samples the conditional vector and corresponding data."""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame,
        output_info: List[ColumnTransformInfo],
        log_frequency: bool = True,
        device: Any = DEVICE,
    ) -> None:
        self._device = device
        self._indices = list(range(len(data)))
        self._num_samples = len(data)

        self._internal_setup(data, output_info, log_frequency=log_frequency)

        self._prepare_train_conditionals()

    def _random_choice_prob_index(self, discrete_column_id: int) -> np.ndarray:
        probs = self._discrete_feat_value_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def get_train_conditionals(self) -> np.ndarray:
        return self._train_conditional

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(self, batch: int, with_ids: bool = False) -> Any:
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

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
    def sample_conditional_indices(
        self,
        cat_feats: np.ndarray,
        cat_values: np.ndarray,
    ) -> List[int]:
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        assert len(cat_values) == len(cat_feats)

        idx = []
        for c, o in zip(cat_feats, cat_values):
            idx.append(np.random.choice(self._categorical_value_to_row_[c][o]))

        return idx

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return self._n_conditional_dimension

    def __iter__(self) -> Generator:
        np.random.shuffle(self._indices)

        for idx in self._indices:
            yield idx

    def __len__(self) -> int:
        return self._num_samples

    def _internal_setup(
        self,
        data: pd.DataFrame,
        output_info: List[ColumnTransformInfo],
        log_frequency: bool,
    ) -> None:
        assert data.shape[1] == sum([item.output_dimensions for item in output_info])

        def is_discrete_column(column_info: ColumnTransformInfo) -> bool:
            return column_info.column_type == "discrete"

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

        assert st == data.shape[1]

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
        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                ed = st + column_info.output_dimensions
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
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
        assert st == data.shape[1]

    def _prepare_train_conditionals(self) -> None:
        self._train_conditional = None
        self._train_mask = None
        if self._n_discrete_columns == 0:
            return

        (
            self._train_conditional,
            categoricals,
            categoricals_vals,
        ) = self.sample_conditional(self._num_samples, with_ids=True)

        self._indices = self.sample_conditional_indices(categoricals, categoricals_vals)

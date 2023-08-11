# stdlib
import itertools
from typing import Any, Optional, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments

# synthcity absolute
from synthcity.utils.constants import DEVICE

# synthcity relative
from .aim import AIM
from .mbi.dataset import Dataset
from .mbi.domain import Domain


class TabularAIM:
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_aim.TabularAIM
    :parts: 1


    Adaptive and Iterative Mechanism (AIM) implementation, based on:
     - code: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py
     - paper: https://www.vldb.org/pvldb/vol15/p2599-mckenna.pdf.


    Args:
        X (pd.DataFrame): Reference dataset, used for training the tabular encoder
        # AIM parameters

        # core plugin arguments
        encoder_max_clusters (int = 20): The max number of clusters to create for continuous columns when encoding with TabularEncoder. Defaults to 20.
        encoder_whitelist (list = []): Ignore columns from encoding with TabularEncoder. Defaults to [].
        device: Union[str, torch.device] = DEVICE, # This is not used for this model, as it is built with sklearn, which is cpu only
        random_state (int, optional): _description_. Defaults to 0. # This is not used for this model
        **kwargs (Any): The keyword arguments are passed to a SKLearn RandomForestClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        # AIM parameters
        epsilon: float = 1.0,
        delta: float = 1e-9,
        max_model_size: int = 80,
        degree: int = 2,
        num_marginals: Optional[int] = None,
        max_cells: int = 1000,
        # core plugin arguments
        encoder_max_clusters: int = 20,
        encoder_whitelist: list = [],
        device: Union[str, torch.device] = DEVICE,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        logging_epoch: int = 100,
        random_state: int = 0,
        **kwargs: Any,
    ):
        super(TabularAIM, self).__init__()
        self.columns = X.columns
        self.epsilon = epsilon
        self.delta = delta
        self.max_model_size = max_model_size
        self.degree = degree
        self.num_marginals = num_marginals
        self.max_cells = max_cells
        self.prng = np.random

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        """

        Args:
            data: Pandas DataFrame that contains the tabular data

        Returns:
            AIMTrainer used for the fine-tuning process
        """
        domain_shapes = X.nunique().to_dict()
        mbi_domain = Domain(self.columns, domain_shapes.values())
        self.dataset = Dataset(X, mbi_domain)

        workload = list(itertools.combinations(self.dataset.domain, self.degree))
        if len(workload) == 0:
            raise ValueError("No workload found. Is the dataset empty?")
        workload = [
            cl for cl in workload if self.dataset.domain.size(cl) <= self.max_cells
        ]
        if len(workload) == 0:
            raise ValueError(
                "Domain sizes for the cells are too large. Increase max_cells values or further discretize the data."
            )
        if self.num_marginals is not None:
            workload = [
                workload[i]
                for i in self.prng.choice(
                    len(workload), self.num_marginals, replace=False
                )
            ]

        self.workload = [(cl, 1.0) for cl in workload]
        self.model = AIM(self.epsilon, self.delta, max_model_size=self.max_model_size)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        start_col: Optional[str] = "",
        start_col_dist: Optional[Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
    ) -> pd.DataFrame:
        """
        Generates tabular data using the trained AIM model.

        Args:
            count (int): The number of samples to generate

        Returns:
            pd.DataFrame: n_samples rows of generated data
        """
        synth_dataset = self.model.run(self.dataset, self.workload)
        return synth_dataset.df

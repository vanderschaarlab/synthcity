# stdlib
from typing import Any, Union

# third party
import pandas as pd
import torch
from arfpy import arf
from pydantic import validate_arguments

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE


class TabularARF:
    def __init__(
        self,
        # ARF parameters
        X: pd.DataFrame,
        num_trees: int = 30,
        delta: int = 0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = True,
        min_node_size: int = 5,
        max_attempts: int = 10,
        # ARF forde parameters
        dist: str = "truncnorm",
        oob: bool = False,
        alpha: float = 0,
        # core plugin arguments
        device: Union[str, torch.device] = DEVICE,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        logging_epoch: int = 100,
        random_state: int = 0,
        **kwargs: Any,
    ):
        """
        .. inheritance-diagram:: synthcity.plugins.core.models.tabular_arf.TabularARF
        :parts: 1


        Adversarial random forests for tabular data.

        This class cis a simple wrapper around the arfpy module which implements Adversarial random forests for tabular data.

        Args:
            # ARF parameters
            X (pd.DataFrame): Reference dataset, used for training the tabular encoder? # TODO: check if this is needed? Delete?
            num_trees (int, optional): Number of trees to grow in each forest. Defaults to 30
            delta (int, optional): Tolerance parameter. Algorithm converges when OOB accuracy is < 0.5 + `delta`. Defaults to 0.
            max_iters (int, optional): Maximum iterations for the adversarial loop. Defaults to 10.
            early_stop (bool, optional): Terminate loop if performance fails to improve from one round to the next?. Defaults to True.
            verbose (bool, optional): Print discriminator accuracy after each round?. Defaults to True.
            min_node_size (int, optional): minimum number of samples in terminal node. Defaults to 5.
            max_attempts (int, optional): Maximum number of attempts to generate new records. Defaults to 10.

            # ARF forde parameters
            dist (str, optional): Distribution to use for density estimation of continuous features. Distributions implemented so far: "truncnorm", defaults to "truncnorm"
            oob (bool, optional): Only use out-of-bag samples for parameter estimation? If `True`, `x` must be the same dataset used to train `arf`, defaults to False
            alpha (float, optional): Optional pseudocount for Laplace smoothing of categorical features. This avoids zero-mass points when test data fall outside the support of training data. Effectively parametrizes a flat Dirichlet prior on multinomial likelihoods, defaults to 0

            # core plugin arguments
            device: Union[str, torch.device] = DEVICE, # This is not used for this model, as it is built with sklearn, which is cpu only
            random_state (int, optional): _description_. Defaults to 0. # This is not used for this model
            **kwargs (Any): The keyword arguments are passed to a SKLearn RandomForestClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
        """
        super(TabularARF, self).__init__()
        self.columns = X.columns
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self.max_attempts = max_attempts

        self.dist = dist
        self.oob = oob
        self.alpha = alpha

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
    ) -> None:
        self.model = arf.arf(
            x=X,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            early_stop=self.early_stop,
            verbose=self.verbose,
            min_node_size=self.min_node_size,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
    ) -> pd.DataFrame:
        self.model.forde(dist=self.dist, oob=self.oob, alpha=self.alpha)
        for i in range(self.max_attempts - 1):
            try:
                samples = self.model.forge(n=count)
                return pd.DataFrame(samples)
            except Exception as e:
                log.debug(f"Attempt {i} failed due to error: {e}. Trying again")
                pass
        samples = self.model.forge(n=count)
        return pd.DataFrame(samples)

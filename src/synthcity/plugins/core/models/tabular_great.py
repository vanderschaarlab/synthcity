# stdlib
from typing import Any, Dict, Optional, Union

# third party
import pandas as pd
import torch
from pydantic import validate_arguments

try:
    # third party
    import be_great
except ImportError:
    raise ImportError(
        """
GReaT is not installed. Please install it with pip install GReaT.
Please be aware that GReaT is only available for python >= 3.9.
"""
    )
# synthcity absolute
from synthcity.utils.constants import DEVICE


class TabularGReaT:
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_great.TabularGReaT
    :parts: 1


    Generation of Realistic Tabular data with pretrained Transformer-based language models (GReaT) implementation.

    This class is a simple wrapper around the be_great module which GReaT.

    Args:
        # GReaT parameters
        X (pd.DataFrame): Reference dataset, used for training the tabular encoder
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        n_iter (int = 100): Number of training iterations to fine-tune the model
        experiment_dir (str = trainer_great):  Directory, where the training checkpoints will be saved
        batch_size (int = 8): Batch size used for fine-tuning
        efficient_finetuning (str): Indication of fune-tuning method
        train_kwargs (dict = {}): Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
            see here the full list of all possible values
            https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

        # core plugin arguments
        encoder_max_clusters (int = 20): The max number of clusters to create for continuous columns when encoding with TabularEncoder. Defaults to 20.
        encoder_whitelist (list = []): Ignore columns from encoding with TabularEncoder. Defaults to [].
        device: Union[str, torch.device] = DEVICE, # This is not used for this model, as it is built with sklearn, which is cpu only
        random_state (int, optional): _description_. Defaults to 0. # This is not used for this model
        **kwargs (Any): The keyword arguments are passed to a SKLearn RandomForestClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
    """

    def __init__(
        self,
        # GReaT parameters
        X: pd.DataFrame,
        llm: str = "distilgpt2",
        n_iter: int = 100,
        experiment_dir: str = "trainer_great",
        batch_size: int = 8,
        train_kwargs: Dict = {},
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
        super(TabularGReaT, self).__init__()
        self.columns = X.columns
        self.llm = llm
        self.n_iter = n_iter
        self.experiment_dir = experiment_dir
        self.batch_size = batch_size
        self.train_kwargs = train_kwargs
        self.device = device

        self.model = be_great.GReaT(
            llm=self.llm,
            experiment_dir=self.experiment_dir,
            epochs=self.n_iter,
            batch_size=self.batch_size,
            **self.train_kwargs,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        conditional_col: Optional[str] = None,
        resume_from_checkpoint: Union[bool, str] = False,
    ) -> Any:
        """
        Wrapper around be_great.GReat.fit(), which Fine-tunes GReaT using tabular data.

        Args:
            data: Pandas DataFrame that contains the tabular data
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature.
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        self.model.fit(
            data=X,
            conditional_col=conditional_col,
            resume_from_checkpoint=resume_from_checkpoint,
        )
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
        Generates tabular data using the trained GReaT model.

        Args:
            count (int): The number of samples to generate
            start_col (Optional[str], optional): Feature to use as a starting point for the generation process.
              If not given, the target learned during fitting is used as starting point. Defaults to "".
            start_col_dist (Optional[Union[dict, list]], optional): Feature distribution of the starting feature.
              Should have the format "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values
              for continuous columns.              If not given, the target distribution learned during the fitting
              is used as starting point. Defaults to None.
            temperature (float, optional): The generation samples each token from the probability distribution given by a softmax
              function. The temperature parameter controls the softmax function. A low temperature makes it sharper
              (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
              See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
              process. Defaults to 0.7.
            k (int, optional): Sampling Batch Size. Set as high as possible. Speeds up the generation process
              significantly. Defaults to 100.
            max_length (int, optional):  Maximal number of tokens to generate - has to be long enough to not cut any information!
              Defaults to 100.

        Returns:
            pd.DataFrame: n_samples rows of generated data
        """
        samples = self.model.sample(
            n_samples=count,
            start_col=start_col,
            start_col_dist=start_col_dist,
            temperature=temperature,
            k=k,
            max_length=max_length,
            device=self.device,
        )
        return pd.DataFrame(samples)

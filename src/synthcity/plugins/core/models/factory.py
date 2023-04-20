# stdlib
from importlib import import_module
from typing import Any, Union

# third party
from pydantic import validate_arguments
from torch import nn
from tsai.models.InceptionTime import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.OmniScaleCNN import OmniScaleCNN
from tsai.models.ResCNN import ResCNN
from tsai.models.RNN_FCN import MLSTM_FCN
from tsai.models.TCN import TCN
from tsai.models.XceptionTime import XceptionTime
from tsai.models.XCM import XCM

# synthcity relative
from .feature_encoder import (
    BayesianGMMEncoder,
    DatetimeEncoder,
    FeatureEncoder,
    GaussianQuantileTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from .layers import GumbelSoftmax

MODELS = dict(
    mlp=".mlp.MLP",
    # rnn models
    rnn=nn.RNN,
    gru=nn.GRU,
    lstm=nn.LSTM,
    # attention models
    transformer=".transformer.TransformerModel",
    tabnet=".tabnet.TabNet",
    # time series models
    inceptiontime=InceptionTime,
    omniscalecnn=OmniScaleCNN,
    rescnn=ResCNN,
    mlstmfcn=MLSTM_FCN,
    tcn=TCN,
    xceptiontime=XceptionTime,
    xcm=XCM,
    inceptiontimeplus=InceptionTimePlus,
)

ACTIVATIONS = dict(
    none=nn.Identity,
    elu=nn.ELU,
    relu=nn.ReLU,
    leakyrelu=nn.LeakyReLU,
    selu=nn.SELU,
    tanh=nn.Tanh,
    sigmoid=nn.Sigmoid,
    softmax=GumbelSoftmax,
    vanilla_softmax=nn.Softmax,
    gelu=nn.GELU,
    silu=nn.SiLU,
    swish=nn.SiLU,
    hardtanh=nn.Hardtanh,
    celu=nn.CELU,
    glu=nn.GLU,
    prelu=nn.PReLU,
    relu6=nn.ReLU6,
    logsigmoid=nn.LogSigmoid,
    logsoftmax=nn.LogSoftmax,
    softplus=nn.Softplus,
)

FEATURE_ENCODERS = dict(
    datetime=DatetimeEncoder,
    onehot=OneHotEncoder,
    ordinal=OrdinalEncoder,
    label=LabelEncoder,
    standard=StandardScaler,
    minmax=MinMaxScaler,
    robust=RobustScaler,
    quantile=GaussianQuantileTransformer,
    bayesiangmm=BayesianGMMEncoder,
    none=FeatureEncoder,
    passthrough=FeatureEncoder,
)


def _get(type_: Union[str, type], params: dict, registry: dict) -> Any:
    if isinstance(type_, type):
        return type_(**params)
    type_ = type_.lower().replace("_", "")
    if type_ in registry:
        cls = registry[type_]
        if isinstance(cls, str):
            cls = registry[type_] = _dynamic_import(cls)
        return cls(**params)
    raise ValueError(f"Unknown type: {type_}")


def _dynamic_import(path: str) -> type:
    """Avoid circular imports by importing dynamically."""
    if path.startswith("."):
        package = __name__.rsplit(".", 1)[0]
    else:
        package = None
    mod_path, name = path.rsplit(".", 1)
    module = import_module(mod_path, package)
    return getattr(module, name)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_model(block: Union[str, type], params: dict) -> Any:
    """Get a model from a name or a class.

    Named models:
    - mlp
    - rnn
    - gru
    - lstm
    - transformer
    - tabnet
    """
    return _get(block, params, MODELS)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_nonlin(nonlin: Union[str, nn.Module], params: dict = {}) -> Any:
    """Get a nonlinearity layer from a name or a class."""
    return _get(nonlin, params, ACTIVATIONS)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_feature_encoder(encoder: Union[str, type], params: dict = {}) -> Any:
    """Get a feature encoder from a name or a class.

    Named encoders:
    - datetime
    - onehot
    - label
    - standard
    - minmax
    - robust
    - quantile
    - bayesian_gmm
    - passthrough
    """
    if isinstance(encoder, type):  # custom encoder
        encoder = FeatureEncoder.wraps(encoder)
    return _get(encoder, params, FEATURE_ENCODERS)

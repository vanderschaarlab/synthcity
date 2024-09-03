# stdlib
import hashlib
from pathlib import Path
from typing import Any, List, Union

# third party
import cloudpickle
import pandas as pd
from opacus import PrivacyEngine

# The list of plugins that are not simply loadable with cloudpickle
unloadable_plugins: List[str] = [
    "dpgan",  # DP-GAN plugin id not loadable with cloudpickle due to the DPOptimizer
]


# TODO: simplify this function back to just cloudpickle.dumps(model), if possible (i.e. if the DPOptimizer is not needed or becomes loadable with cloudpickle)
def save(custom_model: Any) -> bytes:
    """
    Serialize a custom model object that may or may not contain a PyTorch model with a privacy engine.

    Args:
        custom_model: The custom model object to serialize, potentially containing a PyTorch model with a privacy engine.

    Returns:
        bytes: Serialized model state as bytes.
    """
    # Checks is custom model is not a plugin without circular import
    if not hasattr(custom_model, "name"):
        return cloudpickle.dumps(custom_model)

    if custom_model.name() not in unloadable_plugins:
        return cloudpickle.dumps(custom_model)

    # Initialize the checkpoint dictionary
    checkpoint = {
        "custom_model_state": None,
        "pytorch_model_state": None,
        "privacy_engine_state": None,
        "optimizer_state": None,
        "optimizer_class": None,
        "optimizer_defaults": None,
    }

    # Save the state of the custom model object (excluding the PyTorch model and optimizer)
    custom_model_state = {
        key: value for key, value in custom_model.__dict__.items() if key != "model"
    }
    checkpoint["custom_model_state"] = cloudpickle.dumps(custom_model_state)

    # Check if the custom model contains a PyTorch model
    pytorch_model = None
    if hasattr(custom_model, "model"):
        pytorch_model = getattr(custom_model, "model")

    # If a PyTorch model is found, check if it's using Opacus for DP
    if pytorch_model:
        checkpoint["pytorch_model_state"] = pytorch_model.state_dict()
        if hasattr(pytorch_model, "privacy_engine") and isinstance(
            pytorch_model.privacy_engine, PrivacyEngine
        ):
            # Handle DP Optimizer
            optimizer = pytorch_model.privacy_engine.optimizer

            checkpoint.update(
                {
                    "optimizer_state": optimizer.state_dict(),
                    "privacy_engine_state": pytorch_model.privacy_engine.state_dict(),
                    "optimizer_class": optimizer.__class__,
                    "optimizer_defaults": optimizer.defaults,
                }
            )

    # Serialize the entire state with cloudpickle
    return cloudpickle.dumps(checkpoint)


# TODO: simplify this function back to just cloudpickle.loads(model), if possible (i.e. if the DPOptimizer is not needed or becomes loadable with cloudpickle)
def load(buff: bytes, custom_model: Any = None) -> Any:
    """
    Deserialize a custom model object that may or may not contain a PyTorch model with a privacy engine.

    Args:
        buff (bytes): Serialized model state as bytes.
        custom_model: The custom model instance to load the state into.

    Returns:
        custom_model: The deserialized custom model with its original state.
    """
    # Load the checkpoint
    if custom_model is None or custom_model.name() not in unloadable_plugins:
        return cloudpickle.loads(buff)

    if custom_model is None:
        raise ValueError(
            f"custom_model must be provided when loading one of the following plugins: {unloadable_plugins}"
        )

    checkpoint = cloudpickle.loads(buff)
    # Restore the custom model's own state (excluding the PyTorch model)
    custom_model_state = cloudpickle.loads(checkpoint["custom_model_state"])
    for key, value in custom_model_state.items():
        setattr(custom_model, key, value)

    # Find the PyTorch model inside the custom model if it exists
    pytorch_model = None
    if hasattr(custom_model, "model"):
        pytorch_model = getattr(custom_model, "model")

    # Load the states into the PyTorch model if it exists
    if pytorch_model and checkpoint["pytorch_model_state"] is not None:
        pytorch_model.load_state_dict(checkpoint["pytorch_model_state"])

        # Check if the serialized model had a privacy engine
        if checkpoint["privacy_engine_state"] is not None:
            # If there was a privacy engine, recreate and reattach it
            optimizer_class = checkpoint["optimizer_class"]
            optimizer_defaults = checkpoint["optimizer_defaults"]

            # Ensure the optimizer is correctly created with model's parameters
            optimizer = optimizer_class(
                pytorch_model.parameters(), **optimizer_defaults
            )

            # Recreate the privacy engine
            privacy_engine = PrivacyEngine(
                pytorch_model,
                sample_rate=optimizer.defaults.get(
                    "sample_rate", 0.01
                ),  # Use saved or default values
                noise_multiplier=optimizer.defaults.get("noise_multiplier", 1.0),
                max_grad_norm=optimizer.defaults.get("max_grad_norm", 1.0),
            )
            privacy_engine.attach(optimizer)

            # Load the saved states
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            privacy_engine.load_state_dict(checkpoint["privacy_engine_state"])

            # Assign back to the PyTorch model (or the appropriate container)
            pytorch_model.privacy_engine = privacy_engine

    return custom_model


def save_to_file(path: Union[str, Path], model: Any) -> Any:
    path = Path(path)
    ppath = path.absolute().parent

    if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Dataframe hashing, used for caching/backups"""
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))


def dataframe_cols_hash(df: pd.DataFrame) -> str:
    df.columns = df.columns.map(str)
    cols = "--".join(list(sorted(df.columns)))

    return hashlib.sha256(cols.encode()).hexdigest()

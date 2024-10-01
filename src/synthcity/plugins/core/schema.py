# stdlib
from typing import Any, Dict, Generator, List, Optional, Union

# third party
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    validate_arguments,
)

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    DatetimeDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
    PassThroughDistribution,
)


class Schema(BaseModel):
    """
    Utility class for defining the schema of a Dataset.

    Constructor Args:
        domain: Dict
            A dictionary of feature_name: Distribution.
        sampling_strategy: str
            Taking value of "marginal" (default) or "uniform" (for debugging).
        protected_cols: List[str]
            List of columns that are exempt from distributional constraints (e.g. ID column)
        random_state: int
            Random seed (default 0)
        data: Any
            (Optional) the data set
    """

    sampling_strategy: str = Field(default="marginal")
    protected_cols: List[str] = []
    random_state: int = Field(default=0)
    domain: Dict = Field(default_factory=dict)

    data: Optional[Union[DataLoader, pd.DataFrame]] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    def validate_data(cls, v: Any) -> Optional[DataLoader]:
        if v is not None:
            if isinstance(v, pd.DataFrame):
                return GenericDataLoader(v)
            elif isinstance(v, DataLoader):
                return v
            else:
                raise ValueError(
                    f"Invalid data type for 'data': {type(v)}. Expected DataLoader or pandas DataFrame."
                )
        return v

    @model_validator(mode="after")
    def initialize_domain(cls, model: "Schema") -> "Schema":
        if model.data is not None:
            X = model.data.dataframe()
            model.domain = model._infer_domain(
                X,
                sampling_strategy=model.sampling_strategy,
                random_state=model.random_state,
            )
            # Remove 'data' attribute from the model
            del model.__dict__["data"]
            if "data" in model.__fields_set__:
                model.__fields_set__.remove("data")
        return model

    @validate_arguments
    def get(self, feature: str) -> Distribution:
        """Get the Distribution of a feature.

        Args:
            feature: str. the feature name

        Returns:
            The feature distribution
        """
        if feature not in self.domain:
            raise ValueError(f"invalid feature {feature}")

        return self.domain[feature]

    @validate_arguments
    def __getitem__(self, key: str) -> Distribution:
        """Get the Distribution of a feature.

        Args:
            feature: str. the feature name

        Returns:
            The feature distribution
        """
        return self.get(key)

    def __iter__(self) -> Generator:
        """Iterate the features distribution"""
        for x in self.domain:
            yield x

    def __len__(self) -> int:
        """Get the number of features"""
        return len(self.domain)

    def includes(self, other: "Schema") -> bool:
        """Test if another schema is included in the local one."""
        for feature in other:
            if feature in self.protected_cols:
                continue
            if feature not in self.domain:
                return False

            if not self[feature].includes(other[feature]):
                return False

        return True

    def features(self) -> List:
        return list(self.domain.keys())

    def sample(self, count: int) -> pd.DataFrame:
        data = {}
        for col, dist in self.domain.items():
            samples = dist.sample(count)
            data[col] = samples
        return pd.DataFrame(data)

    def adapt_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applying the data type to a new data frame

        Args:
            X: pd.DataFrame
                A new data frame to be adapted.

        Returns:
            A data frame whose data types are coerced to be the same with the Schema.
            If the data frame contains new features, these will be retained as is.
        """
        for feature in self.domain:
            if feature not in X.columns:
                continue
            X[feature] = X[feature].astype(
                self.domain[feature].dtype(), errors="ignore"
            )

        return X

    def as_constraints(self) -> Constraints:
        rules = []
        for feature, dist in self.domain.items():
            rules.extend(dist.as_constraint().rules)
        return Constraints(rules=rules)

    @classmethod
    def from_constraints(cls, constraints: Constraints) -> "Schema":
        domain: Dict = {}
        feature_params: Dict = {}

        # Collect constraint information
        for feature, op, value in constraints.rules:
            if feature not in feature_params:
                feature_params[feature] = {
                    "name": feature,
                    "random_state": None,
                    "low": None,
                    "high": None,
                    "dtype": "float",  # Default to 'float' if not specified
                    "choices": [],
                }

            params = feature_params[feature]

            if op in ["ge", ">="]:
                if params["low"] is None or value > params["low"]:
                    params["low"] = value
            elif op in ["le", "<="]:
                if params["high"] is None or value < params["high"]:
                    params["high"] = value
            elif op in ["eq", "=="]:
                # For '==', set both 'low' and 'high' to value
                params["low"] = value
                params["high"] = value
            elif op in ["in", "isin"]:
                if isinstance(value, list):
                    params["choices"].extend(value)
                else:
                    params["choices"].append(value)
            elif op == "dtype":
                params["dtype"] = value
            else:
                # Handle other operators if necessary
                pass

        # Create distribution objects
        for feature, params in feature_params.items():
            dtype = params["dtype"]
            if dtype == "float":
                if params["low"] is None or params["high"] is None:
                    raise ValueError(
                        f"Cannot create FloatDistribution for '{feature}' without 'low' and 'high' values."
                    )
                domain[feature] = FloatDistribution(
                    name=params["name"],
                    random_state=params["random_state"],
                    low=params["low"],
                    high=params["high"],
                )
            elif dtype == "int":
                if params["low"] is None or params["high"] is None:
                    raise ValueError(
                        f"Cannot create IntegerDistribution for '{feature}' without 'low' and 'high' values."
                    )
                domain[feature] = IntegerDistribution(
                    name=params["name"],
                    random_state=params["random_state"],
                    low=int(params["low"]),
                    high=int(params["high"]),
                    step=1,  # Default step to 1 or adjust as needed
                )
            elif dtype in ["category", "object"]:
                choices = params.get("choices")
                if choices is None or not choices:
                    raise ValueError(
                        f"Cannot create CategoricalDistribution for '{feature}' without 'choices'."
                    )
                domain[feature] = CategoricalDistribution(
                    name=params["name"],
                    random_state=params["random_state"],
                    choices=list(set(choices)),
                )
            else:
                raise ValueError(
                    f"Unsupported dtype '{dtype}' for feature '{feature}'."
                )

        return cls(domain=domain)

    def _infer_domain(
        self,
        X: pd.DataFrame,
        sampling_strategy: str,
        random_state: int,
    ) -> Dict[str, Distribution]:
        feature_domain: Dict[str, Distribution] = {}

        for idx, col in enumerate(X.columns):
            col_random_state = random_state + idx + 1  # Ensure unique seeds

            try:
                if sampling_strategy == "marginal":
                    if col in self.protected_cols:
                        feature_domain[col] = PassThroughDistribution(
                            name=col,
                            data=X[col],
                            random_state=col_random_state,
                        )
                        continue

                    is_categorical = pd.api.types.is_categorical_dtype(X[col])
                    is_object = X[col].dtype == object
                    is_bool = pd.api.types.is_bool_dtype(X[col])
                    is_integer = pd.api.types.is_integer_dtype(X[col])
                    is_float = pd.api.types.is_float_dtype(X[col])
                    is_datetime = pd.api.types.is_datetime64_any_dtype(X[col])

                    if is_categorical or is_object or is_bool:
                        feature_domain[col] = CategoricalDistribution(
                            name=col,
                            data=X[col],
                            random_state=col_random_state,
                        )
                    elif is_integer:
                        feature_domain[col] = IntegerDistribution(
                            name=col,
                            data=X[col],
                            random_state=col_random_state,
                        )
                    elif is_float:
                        feature_domain[col] = FloatDistribution(
                            name=col,
                            data=X[col],
                            random_state=col_random_state,
                        )
                    elif is_datetime:
                        feature_domain[col] = DatetimeDistribution(
                            name=col,
                            data=X[col],
                            random_state=col_random_state,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported data type for column '{col}' with dtype {X[col].dtype}"
                        )
                elif sampling_strategy == "uniform":

                    is_categorical = pd.api.types.is_categorical_dtype(X[col])
                    is_object = X[col].dtype == object
                    is_bool = pd.api.types.is_bool_dtype(X[col])
                    is_integer = pd.api.types.is_integer_dtype(X[col])
                    is_float = pd.api.types.is_float_dtype(X[col])
                    is_datetime = pd.api.types.is_datetime64_any_dtype(X[col])

                    if (
                        pd.api.types.is_categorical_dtype(X[col])
                        or X[col].dtype == object
                        or pd.api.types.is_bool_dtype(X[col])
                    ):
                        feature_domain[col] = CategoricalDistribution(
                            name=col,
                            choices=list(X[col].unique()),
                            random_state=col_random_state,
                            sampling_strategy=sampling_strategy,
                        )
                    elif pd.api.types.is_integer_dtype(X[col]):
                        feature_domain[col] = IntegerDistribution(
                            name=col,
                            low=X[col].min(),
                            high=X[col].max(),
                            random_state=col_random_state,
                            sampling_strategy=sampling_strategy,
                        )
                    elif pd.api.types.is_float_dtype(X[col]):
                        feature_domain[col] = FloatDistribution(
                            name=col,
                            low=X[col].min(),
                            high=X[col].max(),
                            random_state=col_random_state,
                            sampling_strategy=sampling_strategy,
                        )
                    elif pd.api.types.is_datetime64_any_dtype(X[col]):
                        feature_domain[col] = DatetimeDistribution(
                            name=col,
                            low=X[col].min(),
                            high=X[col].max(),
                            random_state=col_random_state,
                            sampling_strategy=sampling_strategy,
                        )
                else:
                    raise ValueError(
                        f"Unsupported sampling strategy '{sampling_strategy}'"
                    )
            except Exception as e:
                log.error(f"Exception occurred while processing column '{col}': {e}")
                raise
        return feature_domain

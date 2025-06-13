# stdlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from synthcity.plugins.core.models.syn_seq.methods import (
    generate_cart,
    generate_ctree,
    generate_lognorm,
    generate_logreg,
    generate_norm,
    generate_pmm,
    generate_polyreg,
    generate_random,
    generate_rf,
    generate_swr,
    syn_cart,
    syn_ctree,
    syn_lognorm,
    syn_logreg,
    syn_norm,
    syn_pmm,
    syn_polyreg,
    syn_random,
    syn_rf,
    syn_swr,
)

METHOD_MAP: Dict[str, Tuple[Any, Any]] = {
    "cart": (syn_cart, generate_cart),
    "ctree": (syn_ctree, generate_ctree),
    "logreg": (syn_logreg, generate_logreg),
    "norm": (syn_norm, generate_norm),
    "pmm": (syn_pmm, generate_pmm),
    "polyreg": (syn_polyreg, generate_polyreg),
    "rf": (syn_rf, generate_rf),
    "lognorm": (syn_lognorm, generate_lognorm),
    "random": (syn_random, generate_random),
    "swr": (syn_swr, generate_swr),
}

NUMERIC_MARKER = -777777777
MISSING_MARKER = -999999999


class Syn_Seq:

    """Synthetic Sequence Generator model.

    This model generates synthetic data sequentially, column by column, using various methods.
    It supports categorical columns with special values and numeric columns, and it can handle
    missing values by using a numeric marker.
    It fits each column based on the metadata provided by the loader and generates synthetic data
    by sampling from the fitted models.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, random_state: int = 0, sampling_patience: int = 100) -> None:
        """
        Args:
            random_state: Random seed.
            sampling_patience: Maximum number of attempts in generation.
        """
        self.random_state = random_state
        self.sampling_patience = sampling_patience
        self.cat_distributions: Dict[str, Dict[Any, float]] = {}
        self._model_trained = False
        self._syn_order: List[str] = []
        self._method_map: Dict[str, str] = {}
        self._varsel: Dict[str, List[str]] = {}
        self._col_models: Dict[str, Optional[Dict[str, Any]]] = {}
        self._first_col_values: Dict[str, np.ndarray] = {}

    def fit_col(
        self,
        loader: Any,
        label_encoder: Any,
        loader_info: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "Syn_Seq":
        """
        Fit each column sequentially using metadata from the loader.
        For each _cat column in the training data, record its full distribution
        (casting values to int) and record the list of special values. Then, for base
        columns with special values, filter out rows whose value is special so that the
        model sees only numeric values.
        """
        info_dict = loader_info
        training_data = loader.dataframe().copy()
        if training_data.empty:
            raise ValueError("No data => cannot fit Syn_Seq aggregator")

        print(info_dict)
        self._syn_order = info_dict.get("syn_order", list(training_data.columns))
        self._method_map = info_dict.get("method", {})
        self._varsel = info_dict.get("variable_selection", {})

        for col in training_data.columns:
            if col.endswith("_cat"):
                value_counts = (
                    training_data[col].astype(int).value_counts(normalize=True)
                )
                self.cat_distributions[col] = value_counts.to_dict()

        for col in self._syn_order:
            if col.endswith("_cat"):
                self._method_map[col] = "cart"
                base_col = col[:-4]
                if base_col in self._varsel:
                    self._varsel[col] = self._varsel[base_col]
                else:
                    idx = self._syn_order.index(col)
                    self._varsel[col] = self._syn_order[:idx]

        print("[INFO] Syn_Seq aggregator: fitting columns...")

        first_col = self._syn_order[0]
        self._first_col_values[first_col] = training_data[first_col].dropna().values
        print(f"Fitting '{first_col}' => stored values from real data. Done.")

        np.random.seed(self.random_state)
        for i, col in enumerate(self._syn_order[1:], start=1):
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(col, self._syn_order[:i])
            y = training_data[col].values
            X = training_data[preds_list].values
            cat_col = col + "_cat"
            if cat_col in preds_list:
                numeric_indices = np.where(
                    label_encoder[cat_col].classes_ == NUMERIC_MARKER
                )[0]

                missing_indices = []
                if MISSING_MARKER in label_encoder[cat_col].classes_:
                    missing_indices = np.where(
                        label_encoder[cat_col].classes_ == MISSING_MARKER
                    )[0]
                if len(numeric_indices) == 0:
                    raise ValueError(
                        f"Numeric marker {NUMERIC_MARKER} not found in {cat_col} classes"
                    )
                numeric_label = numeric_indices[0]
                missing_label = missing_indices[0] if len(missing_indices) > 0 else None
                mask = training_data[cat_col] == numeric_label
                if missing_label is not None:
                    mask &= training_data[cat_col] != missing_label
                y = training_data.loc[mask, col].values
                X = training_data.loc[mask, preds_list].values

            print(f"Fitting '{col}' with '{method_name}' ... ", end="", flush=True)
            try:
                self._col_models[col] = self._fit_single_col(method_name, X, y)
            except Exception as e:
                print(f"Error fitting column {col} with {method_name}: {e}.", end=" ")
                self._col_models[col] = None

            print("Done!")
        self._model_trained = True
        return self

    def _fit_single_col(
        self, method_name: str, X: np.ndarray, y: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Fit a single column using the specified method.
        """
        if len(y) == 0:
            warnings.warn(
                "No training data available for this column! Model will be None"
            )
            return None

        fit_func, _ = METHOD_MAP[method_name]
        try:
            model = fit_func(y, X, random_state=self.random_state)
            return {"name": method_name, "fitted_model": model}
        except Exception as e:
            warnings.warn(f"Failed to fit column with method {method_name}: {str(e)}")
            return None

    def generate_col(self, count: int, label_encoder: Any) -> pd.DataFrame:
        """
        Generate `count` rows sequentially.

        For each base column that has a corresponding _cat distribution, we first generate the
        base column using its fitted model. Then we sample a synthetic _cat indicator (using the
        saved full distribution). For rows where the synthetic indicator is not equal to the numeric
        marker, we override the generated base column value with the special value.
        """
        if not self._model_trained:
            raise RuntimeError("Syn_Seq aggregator not yet fitted")
        if count <= 0:
            return pd.DataFrame({col: [] for col in self._syn_order})

        gen_df = pd.DataFrame({col: [np.nan] * count for col in self._syn_order})

        first_col = self._syn_order[0]
        if (
            first_col in self._first_col_values
            and len(self._first_col_values[first_col]) > 0
        ):
            gen_df[first_col] = np.random.choice(
                self._first_col_values[first_col], size=count, replace=True
            )
        else:
            print("Error generating first column, storing zeroes.", end=" ")
            gen_df[first_col] = 0
        print(f"Generating '{first_col}' => done.")

        for col in self._syn_order[1:]:
            method_name = self._method_map.get(col, "cart")
            preds_list = self._varsel.get(
                col, self._syn_order[: self._syn_order.index(col)]
            )
            cat_col = col + "_cat"
            if cat_col in preds_list:
                print(
                    "Encoded labels for {}:".format(cat_col),
                    label_encoder[cat_col].classes_,
                )
                numeric_indices = np.where(
                    label_encoder[cat_col].classes_ == NUMERIC_MARKER
                )[0]
                if len(numeric_indices) == 0:
                    raise ValueError(
                        f"Numeric marker {NUMERIC_MARKER} not found in {cat_col} classes"
                    )
                numeric_label = numeric_indices[0]
                mask = gen_df[cat_col] == numeric_label
                if mask.sum() > 0:
                    Xsyn_numeric = gen_df.loc[mask, preds_list].values
                    ysyn_numeric = self._generate_single_col(
                        method_name, Xsyn_numeric, col
                    )
                    gen_df.loc[mask, col] = ysyn_numeric
                if (~mask).sum() > 0:
                    special_values = gen_df.loc[~mask, cat_col].map(
                        lambda x: label_encoder[cat_col].classes_[x]
                    )
                    gen_df.loc[~mask, col] = special_values
            else:
                Xsyn = gen_df[preds_list].values
                ysyn = self._generate_single_col(method_name, Xsyn, col)
                gen_df[col] = ysyn
            print(f"Generating '{col}' => done.")

            if col in self.cat_distributions:
                numeric_indices = np.where(
                    label_encoder[col].classes_ == NUMERIC_MARKER
                )[0]
                if len(numeric_indices) > 0:
                    numeric_label = numeric_indices[0].astype(int)
                    if numeric_label not in gen_df[col].unique():
                        print(
                            f"{col} does not contain indicator for numeric values. \n - Model might have failed to fit the data due to highly skewed distribution. \n - Using empirical distribution for generation..."
                        )
                        cat_dist = {
                            int(k): v for k, v in self.cat_distributions[col].items()
                        }
                        gen_df[col] = np.random.choice(
                            list(cat_dist.keys()), size=count, p=list(cat_dist.values())
                        )
        return gen_df

    def _generate_single_col(
        self, method_name: str, Xsyn: np.ndarray, col: str
    ) -> np.ndarray:
        """
        Generate synthetic values for a single column using the fitted model.
        If no model is available for the column, a RuntimeError is raised.
        """
        fit_info = self._col_models.get(col)
        if fit_info is None:
            raise RuntimeError(f"No model available for column {col}.")
        _, generate_func = METHOD_MAP[fit_info["name"]]
        return generate_func(fit_info["fitted_model"], Xsyn)

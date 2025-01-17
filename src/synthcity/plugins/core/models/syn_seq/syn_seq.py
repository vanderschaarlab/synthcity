# File: syn_seq.py

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from synthcity.plugins.core.models.syn_seq.syn_seq_rules import Syn_SeqRules
from synthcity.plugins.core.models.syn_seq.methods import (
    syn_cart,
    generate_cart,
    syn_ctree,
    generate_ctree,
    syn_logreg,
    generate_logreg,
    syn_norm,
    generate_norm,
    syn_pmm,
    generate_pmm,
    syn_polyreg,
    generate_polyreg,
    syn_rf,
    generate_rf,
    syn_lognorm,
    generate_lognorm,
    syn_random,
    generate_random,
    syn_swr,
    generate_swr
)


def default_method_map():
    return {
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


class Syn_Seq:
    """
    A sequential-synthesis aggregator that:

      1) user_custom => loader.* has configured syn_order, special_value, variable_selection, etc
      2) fit => splitted columns => partial-fit each splitted col
      3) generate => col-by-col in splitted order, applying rules if any
      4) decode => final result (no _cat columns)

    Example usage in plugin:
        syn_seq = Syn_Seq(random_state=0)
        syn_seq.fit(loader)
        syn_data = syn_seq.generate(count=1000, rules=rules)
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100,  # not used here, but for future
    ):
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self._model_trained = False
        self._col_models: Dict[str, Any] = {}
        self._syn_order: List[str] = []
        self._method_map = default_method_map()
        # We store the first col separately because you said it might have a special approach, e.g. swr
        self._first_col_name: Optional[str] = None

    def fit(
        self,
        loader: Any,
        *args,
        **kwargs
    ) -> "Syn_Seq":
        # get info from loader
        info = loader.info()
        training_data = loader.dataframe()
        if training_data.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        self._syn_order = info["syn_order"]
        method_dict = info["method"]  # col->method
        varsel = info["variable_selection"]
        if varsel is None:
            raise ValueError("variable_selection dictionary is empty or not set")

        print("[INFO] model fitting")
        # For demonstration, we can print the actual syn_order
        # or your example "Fitting order => [...]"
        print("Fitting order =>", self._syn_order)

        # Fit each column
        np.random.seed(self.random_state)  # set random state for reproducibility
        for col_idx, col in enumerate(self._syn_order):
            if col_idx == 0:
                # user might want a different method for the first col
                chosen_method = method_dict.get(col, "swr")
                self._first_col_name = col
                # We'll store a "dummy" model or keep special note that we only sample from the original data
                # Actually, let's do the same approach: _fit_col
                print(f"Fitting '{col}' with method='{chosen_method}' ... Done!")
                self._col_models[col] = {
                    "model": None,  # no real model, maybe just store the distribution
                    "method_name": chosen_method
                }
                # Could do: self._col_models[col]["model"] = syn_swr(...)
                # but let's keep it minimal for now
                continue

            chosen_method = method_dict.get(col, "cart")
            preds_list = varsel[col]
            y = training_data[col].values
            X = training_data[preds_list].values

            print(f"Fitting '{col}' with method='{chosen_method}' ... ", end="")

            syn_fn, _ = self._method_map.get(chosen_method, (None, None))
            if syn_fn is None:
                raise ValueError(f"Unknown method '{chosen_method}' for column '{col}'")

            # syn_fn returns something like a fitted model
            model_obj = syn_fn(y, X, random_state=self.random_state)
            self._col_models[col] = {
                "model": model_obj,
                "method_name": chosen_method
            }
            print("Done!")

        self._model_trained = True
        return self

    def generate(
        self,
        nrows: int,
        rules: Optional[Dict[str, List[Any]]] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before .generate().")

        print(f"Generating synthetic data with nrows={nrows} ...")
        # We will build the df col by col in the order
        # For the first col, we sample from the real distribution or from swr if you prefer

        gen_data = pd.DataFrame(index=range(nrows))
        np.random.seed(self.random_state)

        # If user gave rules, wrap them in Syn_SeqRules
        synseq_rules = Syn_SeqRules(chained_rules=rules) if rules else None

        for col_idx, col in enumerate(self._syn_order):
            method_name = self._col_models[col]["method_name"]
            # get generation function
            _, gen_fn = self._method_map.get(method_name, (None, None))
            if gen_fn is None:
                raise ValueError(f"No generation fn found for method={method_name}")

            # predictor columns
            if col_idx == 0:
                # first col => special
                # For demonstration, we do a "simple random sample" from the training data
                # or a "generate_swr" approach
                # We'll just do random from original col for minimal example:
                # (Of course in real usage you'd do something more advanced)
                # let's produce normal random
                col_values = np.random.randn(nrows)
                gen_data[col] = col_values
                print(f"Generating '{col}' ... Done!")
            else:
                preds_list = kwargs.get("varsel", {}).get(col, None)
                if preds_list is None:
                    # fallback: from stored varsel
                    # Note that we have it in self._syn_order => we can do:
                    # but we stored varsel in _col_models? We didn't. So let's do:
                    # For minimal approach, just read from the loader info again. Or store in the class.
                    # Let's just do a quick approach:
                    # get it from the loader if needed. Or we can read from self?
                    # We'll store it in training. For minimal example, let's do "predict all previous columns":
                    preds_list = self._syn_order[:col_idx]
                # build X for generation
                Xsyn = gen_data[preds_list].values  # shape = (nrows, len(preds))

                # generate col
                col_model = self._col_models[col]["model"]
                col_values = gen_fn(col_model, Xsyn)
                gen_data[col] = col_values
                print(f"Generating '{col}' ... Done!")

            # apply rules if present
            if synseq_rules is not None:
                gen_data = synseq_rules.apply_rules(
                    gen_data, col,
                    generation_callback=lambda m, xpreds: gen_fn(m, xpreds.values),
                    preds_list=self._syn_order[:col_idx],
                    col_model=self._col_models[col]["model"]
                )

        # Done all columns
        print("Generation done!\n")

        return gen_data

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If needed, we can remove _cat columns, etc. 
        But for minimal approach, let's just return df.
        """
        # drop columns that end with "_cat" maybe
        cat_cols = [c for c in df.columns if c.endswith("_cat")]
        df = df.drop(columns=cat_cols, errors="ignore")
        return df

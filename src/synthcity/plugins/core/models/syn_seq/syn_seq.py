# File: syn_seq.py

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from synthcity.plugins.core.models.syn_seq.syn_seq_rules import Syn_SeqRules
from synthcity.plugins.core.models.syn_seq.methods import (
    syn_cart,
    syn_ctree,
    syn_logreg,
    syn_norm,
    syn_lognorm,
    syn_pmm,
    syn_polyreg,
    syn_rf,
    syn_random,
    syn_swr,
)

class Syn_Seq:
    """
    A sequential-synthesis aggregator that:

      1) user_custom => loader.update_user_custom(user_custom)
      2) encode => splitted columns => partial-fit each splitted col
      3) generate => col-by-col in splitted order
      4) decode => final result (no _cat columns)
    """

    def __init__(
        self,
        random_state: int = 0,
        strict: bool = True,
        sampling_patience: int = 100,
    ):
        self.random_state = random_state
        self.strict = strict
        self.sampling_patience = sampling_patience
        self._model_trained = False
        self._first_col = None

    def fit(
        self,
        loader: Any,  
        *args,
        **kwargs
    ) -> "Syn_Seq":
        
        info = loader.info
        training_data = loader.dataframe()

        if training_data.empty:
            raise ValueError("No data after encoding => cannot train on empty DataFrame.")

        syn_order = info["syn_order"]
        # user might have updated the method via user_custom => stored in loader._method
        method = info["method"]
        varsel = info["variable_selection"]
        if varsel is None:
            raise ValueError("variable selection dictionary is empty")

        print("[INFO] model fitting")
        print("Fitting order =>", syn_order)

        self._col_models = {}
        for col in syn_order:
            if col == syn_order[0]:
                chosen_m = "swr"
                self._first_col = training_data[col]
                continue
            chosen_m = method[col]
            preds_list = varsel[col]
            y = training_data[col].values
            X = training_data[preds_list].values

            print(f"Fitting '{col}' with method='{chosen_m}' ... ", end="")

            _col_models[col] = self._fit_col(y, X, chosen_m, "other hyperparameters for fitting the model")
            print("Done!")

        self._model_trained = True
        return self

    def _fit_col(self, y: np.ndarray, X: np.ndarray, method_name: str,  "other hyperparameters for fitting the model") -> Dict[str, Any]:
        # This function returns dictionary of column name and fitted model. We will use the fitted model to generate the new data.

    def generate(
        self,
        nrows: int,
        rules: Optional[Dict[str, List[Any]]] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        if not self._model_trained:
            raise RuntimeError("Must fit(...) aggregator before .generate().")
        
        col_with_rules = rules.key()
        generated_data = pd.DataFrame
        generated_data = self._first_col.sample(nrows)

        if rules:
            rules = Syn_SeqRules(chained_rules=rules)
        for col in syn_order:
            if col == syn_order[0]:
                continue

            chosen_m = method[col]
            preds_list = varsel[col]
            X_syn = generated_data[preds_list].values
            # y_syn is generated with chosen_m, model from _col_models[col] and synthesized value of that model with syn_"method"(X_syn). This means X_syn has same length as y_syn.
            generated_data[col] = y_syn

            if col in col_with_rules:
                i = 0
                while  i < 10:
                # generated_data[generated_data[rule if] & generated_data[rule then]][col] = nan
                # y_syn generated again within generated_data[generated_data[rule if] & generated_data[rule then]].
                    i += 1
                if generated_data[col].isna() == True:
                    print("Could not generate some values to satisfy the rules. Please modify or remove rules if you don't want nan values")
                 

            return self.decode(pd.dataframe(generated_data))

    # Other helpers




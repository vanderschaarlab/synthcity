# File: plugin_syn_seq.py

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd

from pydantic import validate_arguments

from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.dataloader import (
    DataLoader,
    Syn_SeqDataLoader,
    create_from_info,
)
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.distribution import constraint_to_distribution
from synthcity.plugins.core.schema import Schema
from synthcity.utils.reproducibility import enable_reproducible_results

# aggregator from syn_seq.py
from synthcity.plugins.core.models.syn_seq.syn_seq import Syn_Seq


class Syn_SeqPlugin(Plugin):
    """
    A plugin wrapping the 'Syn_Seq' aggregator in the synthcity Plugin interface.

    Steps:
      1) In .fit(), if the user passes a DataFrame, we wrap it in Syn_SeqDataLoader, then encode the data.
      2) We keep separate:
          - self._orig_schema => schema from the original data
          - self._enc_schema => schema from the encoded data
      3) The aggregator trains column-by-column on the encoded data.
      4) In .generate(), we re-check constraints (including user constraints) referencing
         the original schema. Then we decode back to the original DataFrame structure.

      Additional note: We add `_remap_special_value_rules` to handle user rules referencing
      special values that belong in a 'cat' column, e.g. if user writes
      ("NUM_CIGAR", "<", 0) but that data is actually in "NUM_CIGAR_cat".
    """

    @staticmethod
    def name() -> str:
        return "syn_seq"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List:
        # No tunable hyperparameters here
        return []

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sampling_patience: int = 100,
        strict: bool = True,
        random_state: int = 0,
        compress_dataset: bool = False,
        sampling_strategy: str = "marginal",
        **kwargs: Any
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            strict=strict,
            compress_dataset=compress_dataset,
            sampling_strategy=sampling_strategy,
        )
        # Two separate schema references
        self._orig_schema: Optional[Schema] = None
        self._enc_schema: Optional[Schema] = None
        self._data_info: Optional[Dict] = None
        self._enc_data_info: Optional[Dict] = None

        self.model: Optional[Syn_Seq] = None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: Union[DataLoader, pd.DataFrame], *args: Any, **kwargs: Any) -> Any:
        """
        Wrap a plain DataFrame into Syn_SeqDataLoader if needed, then encode the data.
        Build up:
          - self._orig_schema from the original data
          - self._enc_schema from the encoded data
        Then train the aggregator column-by-column on encoded data.
        """
        # If plain DataFrame, wrap in Syn_SeqDataLoader
        if isinstance(X, pd.DataFrame):
            X = Syn_SeqDataLoader(X)

        if "cond" in kwargs and kwargs["cond"] is not None:
            self.expecting_conditional = True

        enable_reproducible_results(self.random_state)
        self._data_info = X.info()

        # Build schema for the original data
        self._orig_schema = Schema(
            data=X,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        # Encode the data
        X_encoded, self._enc_data_info = X.encode()
        # Build a schema from the encoded data
        self._enc_schema = Schema(
            data=X_encoded,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        # aggregator training
        self.model = Syn_Seq(
            random_state=self.random_state,
            strict=self.strict,
            sampling_patience=self.sampling_patience,
        )
        self.model.fit_col(X_encoded, *args, **kwargs)
        self.fitted = True
        return self

    def training_schema(self) -> Schema:
        """Return the *encoded* schema used for aggregator training."""
        if self._enc_schema is None:
            raise RuntimeError("No encoded schema found. Fit the model first.")
        return self._enc_schema

    def schema(self) -> Schema:
        """Return the original data's schema."""
        if self._orig_schema is None:
            raise RuntimeError("No original schema found. Fit the model first.")
        return self._orig_schema

    def _remap_special_value_rules(
        self,
        rules_dict: Dict[str, List[Tuple[str, str, Any]]],
        enc_schema: Schema
    ) -> Dict[str, List[Tuple[str, str, Any]]]:
        """
        If user wrote rules referencing special values (like -8) on numeric columns,
        we switch them to the corresponding _cat column. For example:
            - "NUM_CIGAR" has special_value list [-8].
            - user sets rule (NUM_CIGAR, "<", 0).
            => Actually these negative codes are stored in "NUM_CIGAR_cat".
            So we redirect the rule to "NUM_CIGAR_cat".
        """
        if not rules_dict:
            return rules_dict

        # gather the special_value map from the *encoded* info
        # (we assume `_enc_data_info["special_value"]` has your col->list).
        special_map = {}
        if self._enc_data_info and "special_value" in self._enc_data_info:
            special_map = self._enc_data_info["special_value"]

        # build base->cat mapping
        base_to_cat = {}
        for col in enc_schema.domain:
            if col.endswith("_cat"):
                base_col = col[:-4]
                base_to_cat[base_col] = col

        new_rules = {}
        for target_col, cond_list in rules_dict.items():
            actual_target_col = target_col
            # If the target_col is known to have special values => rename
            if target_col in special_map and target_col in base_to_cat:
                actual_target_col = base_to_cat[target_col]

            new_cond_list = []
            for (feat_col, op, val) in cond_list:
                new_feat = feat_col
                if feat_col in special_map and feat_col in base_to_cat:
                    # if val is in special_map[feat_col], direct to cat
                    # or if user is comparing negative codes, etc.
                    if any(v == val for v in special_map[feat_col]):
                        new_feat = base_to_cat[feat_col]
                new_cond_list.append((new_feat, op, val))

            new_rules[actual_target_col] = new_cond_list

        return new_rules

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "Syn_SeqPlugin":
        # we do not directly use this `_fit`, it’s overshadowed by .fit() above
        raise NotImplementedError("Use .fit()")

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: Optional[int] = None,
        constraints: Optional[Constraints] = None,
        rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        Generate synthetic data from aggregator.

        1) Combine constraints from original schema with user constraints
        2) Possibly remap user rules -> cat columns for special_value
        3) aggregator -> encoded DataFrame
        4) decode back to original
        5) final constraints check using original schema
        """
        if not self.fitted:
            raise RuntimeError("Must .fit() plugin before calling .generate()")
        if self._orig_schema is None or self._enc_schema is None:
            raise RuntimeError("No schema found. Fit the model first.")

        if random_state is not None:
            enable_reproducible_results(random_state)

        if count is None:
            if self._data_info is not None:
                count = self._data_info["len"]
            else:
                raise ValueError("Cannot determine 'count' for generation")

        has_gen_cond = ("cond" in kwargs) and (kwargs["cond"] is not None)
        if has_gen_cond and not self.expecting_conditional:
            raise RuntimeError(
                "Got generation conditional, but aggregator wasn't trained conditionally"
            )

        # Combine constraints from the original schema
        gen_constraints = self._orig_schema.as_constraints()
        if constraints is not None:
            gen_constraints = gen_constraints.extend(constraints)

        # aggregator generation on encoded schema
        data_syn = self._generate(count, gen_constraints, rules=rules, **kwargs)

        # decode from the encoded data back to original
        data_syn = data_syn.decode()

        # final constraints check using the *original* schema
        final_constraints = self._orig_schema.as_constraints()
        if constraints is not None:
            final_constraints = final_constraints.extend(constraints)

        # If strict, keep only valid rows
        if not data_syn.satisfies(final_constraints) and self.strict:
            data_syn = data_syn.match(final_constraints)

        return data_syn

    def _generate(
        self,
        count: int,
        gen_constraints: Constraints,
        rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        1) Possibly remap user rules for special_value -> cat columns
        2) aggregator => produce an *encoded* DataFrame
        3) Force columns' dtypes (encoded schema)
        """
        if not self.model:
            raise RuntimeError("No aggregator model found")

        # Remap user rules to handle special values in _cat
        if rules is not None:
            rules = self._remap_special_value_rules(rules, self._enc_schema)

        # aggregator generate
        df_syn = self.model.generate_col(count, rules=rules, max_iter_rules=10)
        # Ensure correct dtypes (encoded)
        df_syn = self._enc_schema.adapt_dtypes(df_syn)

        # Return as DataLoader with the *encoded* info
        return create_from_info(df_syn, self._enc_data_info)

plugin = Syn_SeqPlugin

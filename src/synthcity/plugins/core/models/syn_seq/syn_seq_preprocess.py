# syn_seq_preprocess.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple


class SynSeqPreprocessor:
    """
    A class to perform preprocessing and postprocessing for syn_seq.

    Preprocessing:
      - Records the original dtypes.
      - Automatically assigns dtypes (date/category/numeric) when not provided.
      - Converts date columns to datetime and category columns to 'category' dtype.
      - For numeric columns with special values (user_special_values), creates a new 
        categorical column (named base_col_cat) that marks special values:
           * If the value is in the special list, the cell is mapped to the special value.
           * Otherwise, a numeric marker (set to len(specials)) is used.
      
    Postprocessing:
      - Merges back the split (base_col, base_col_cat) columns:
            If the base column is NaN and the corresponding _cat value is one of the special values,
            then the base column is replaced with that special value.
      - Optionally applies user-provided rules sequentially to filter rows.
    """

    def __init__(
        self,
        user_dtypes: Optional[Dict[str, str]] = None,
        user_special_values: Optional[Dict[str, List[Any]]] = None,
        max_categories: int = 20,
    ):
        """
        Args:
            user_dtypes: {col: "date"/"category"/"numeric"}, if not provided, auto-detected.
            user_special_values: {col: [special_value1, special_value2, ...]}
            max_categories: When auto-detecting dtypes, if nunique <= max_categories, assign 'category', else 'numeric'.
        """
        self.user_dtypes = user_dtypes or {}
        self.user_special_values = user_special_values or {}
        self.max_categories = max_categories

        # Internal storage
        self.original_dtypes: Dict[str, str] = {}  # {col: original_dtype}
        self.split_map: Dict[str, str] = {}        # {base_col -> cat_col}
        self.detected_specials: Dict[str, List[Any]] = {}  # user special values

    # =========================================================================
    # PREPROCESS
    # =========================================================================
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the DataFrame.
          1) Record original dtypes.
          2) Auto-assign or apply user-specified dtypes.
          3) Convert date and category columns appropriately.
          4) For numeric columns with special values, create a _cat column.
        """
        df = df.copy()

        # (a) Record original dtypes.
        self._record_original_dtypes(df)

        # (b) Auto-assign dtypes for columns not specified in user_dtypes.
        self._auto_assign_dtypes(df)

        # (c) Apply the specified dtypes.
        self._apply_user_dtypes(df)

        # (d) Split numeric columns that have special values into (base_col, base_col_cat).
        self._split_numeric_columns(df)

        return df

    def _record_original_dtypes(self, df: pd.DataFrame):
        for col in df.columns:
            self.original_dtypes[col] = str(df[col].dtype)

    def _auto_assign_dtypes(self, df: pd.DataFrame):
        """
        For columns not specified in user_dtypes, assign:
          - 'date' if the column is a datetime type.
          - 'category' if nunique <= max_categories.
          - Otherwise, 'numeric'.
        """
        for col in df.columns:
            if col in self.user_dtypes:
                continue

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.user_dtypes[col] = "date"
                print(f"[auto_assign] {col} -> date")
                continue

            nuniq = df[col].nunique(dropna=False)
            if nuniq <= self.max_categories:
                self.user_dtypes[col] = "category"
                print(f"[auto_assign] {col} -> category (nuniq={nuniq})")
            else:
                self.user_dtypes[col] = "numeric"
                print(f"[auto_assign] {col} -> numeric (nuniq={nuniq})")

    def _apply_user_dtypes(self, df: pd.DataFrame):
        """
        Apply the user-specified or auto-assigned dtypes:
          - Convert 'date' columns with pd.to_datetime.
          - Convert 'category' columns with astype('category').
          - Leave 'numeric' columns unchanged.
        """
        for col, dtype_str in self.user_dtypes.items():
            if col not in df.columns:
                continue

            if dtype_str == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype_str == "category":
                df[col] = df[col].astype("category")
            # numeric: no conversion

    def _split_numeric_columns(self, df: pd.DataFrame):
        """
        For each column in user_special_values:
          - Create a new categorical column (base_col_cat) that reflects special values.
          - For each value in the base column:
                If NaN -> return NaN.
                If in specials -> return the special value.
                Otherwise -> return len(specials) (a marker indicating "normal").
        """
        for col, specials in self.user_special_values.items():
            if col not in df.columns:
                continue

            cat_col = col + "_cat"
            self.split_map[col] = cat_col
            self.detected_specials[col] = specials

            # Remove existing cat_col if exists.
            if cat_col in df.columns:
                df.drop(columns=[cat_col], inplace=True)
            base_idx = df.columns.get_loc(col)
            df.insert(base_idx, cat_col, None)

            def cat_mapper(x, specials, normal_marker=None, missing_marker="NAN"):
                if normal_marker is None:
                    normal_marker = "NUMERIC"
                if pd.isna(x):
                    return missing_marker
                elif x in specials:
                    return str(x)
                else:
                    return normal_marker


            df[cat_col] = df[col].apply(lambda x: cat_mapper(x, specials)).astype("category")

    # =========================================================================
    # POSTPROCESS
    # =========================================================================
    def postprocess(self, df: pd.DataFrame, rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None) -> pd.DataFrame:
        """
        Postprocesses the synthetic DataFrame:
         1) Merge back split columns (base_col, base_col_cat) by replacing NaNs in the base column
            with the corresponding special value from the _cat column.
         2) Apply user-provided rules sequentially to filter rows.
        (Note: Date offset restoration is not performed.)
        """
        df = df.copy()
        # Merge split columns.
        df = self._merge_splitted_cols(df)
        # Apply rules if provided.
        if rules is not None:
            df = self.apply_rules(df, rules)
        return df

    def _merge_splitted_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each (base_col, cat_col) pair in split_map, if a base column cell is special values,
        check the corresponding cell in the cat_col.
        If cat_col has "NUMERIC", leave the base_col as it is. If cat_col has "NAN", 
        Then drop the cat_col.
        """

        # Need a logic here

        return df

    def apply_rules(self, df: pd.DataFrame, rules: Dict[str, List[Tuple[str, str, Any]]]) -> pd.DataFrame:
        """
        Apply rules at postprocessing by iteratively dropping rows that do not satisfy each rule,
        in the order provided by the user's input.

        Args:
            df: The synthetic DataFrame.
            rules: A dictionary where each key is a target column and the value is a list of rules
                   in the form (col_feat, operator, value).

        Returns:
            A new DataFrame with rows not satisfying the rules dropped.
        """
        for target_col, rule_list in rules.items():
            for (col_feat, operator, rule_val) in rule_list:
                if col_feat not in df.columns:
                    continue
                if operator in ["=", "=="]:
                    cond = (df[col_feat] == rule_val) | df[col_feat].isna()
                elif operator == ">":
                    cond = (df[col_feat] > rule_val) | df[col_feat].isna()
                elif operator == ">=":
                    cond = (df[col_feat] >= rule_val) | df[col_feat].isna()
                elif operator == "<":
                    cond = (df[col_feat] < rule_val) | df[col_feat].isna()
                elif operator == "<=":
                    cond = (df[col_feat] <= rule_val) | df[col_feat].isna()
                else:
                    cond = pd.Series(True, index=df.index)
                df = df.loc[cond].copy()
        return df

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
      - For numeric columns, if one value accounts for ≥90% of non‐null rows, that value is
        automatically marked as a special value. For each such column, a new categorical column
        (named base_col_cat) is created:
           * If the cell value is missing, it is mapped to the missing marker (here, -999999999).
           * If the cell value equals a detected (or user‐specified) special value, it is left as its
             original numeric value.
           * Otherwise, it is marked with the “numeric” marker (here, -777777777) indicating that the
             value is not special.
      
    Postprocessing:
      - Merges back the split (base_col, base_col_cat) columns:
            For rows where the base column is NaN and the corresponding _cat column 
            is not equal to the numeric marker, the base column is replaced 
            with that special value. In particular, if _cat equals the missing marker,
            the base column is set to np.nan.
      - Optionally applies user‐provided rules sequentially to filter rows.
      - Restores the original column order and dtypes.
    """

    # Define marker constants (choose values unlikely to appear in your data)
    NUMERIC_MARKER = -777777777   # Indicates a normal (non‐special) value
    MISSING_MARKER = -999999999    # Indicates a missing value

    def __init__(
        self,
        user_dtypes: Optional[Dict[str, str]] = None,
        user_special_values: Optional[Dict[str, List[Any]]] = None,
        max_categories: int = 20,
    ):
        """
        Args:
            user_dtypes: {col: "date"/"category"/"numeric"}, if not provided, auto-detected.
            user_special_values: {col: [special_value1, special_value2, ...]}.
                Even if not provided, special values are detected automatically for imbalanced numeric columns.
            max_categories: When auto-detecting dtypes, if nunique <= max_categories, assign 'category', else 'numeric'.
        """
        self.user_dtypes = user_dtypes or {}
        self.user_special_values = user_special_values or {}
        self.max_categories = max_categories

        # Internal storage
        self.original_dtypes: Dict[str, str] = {}       # {col: original_dtype}
        self.split_map: Dict[str, str] = {}             # {base_col -> cat_col}
        self.detected_specials: Dict[str, List[Any]] = {} # stores the special values (detected or user-provided)
        self.dominant_values: Dict[str, Any]

    # =========================================================================
    # PREPROCESSING
    # =========================================================================
    def preprocess(self, df: pd.DataFrame, oversample = False) -> pd.DataFrame:
        """
        Preprocesses the DataFrame:
          1) Records original dtypes.
          2) Auto-assigns or applies user-specified dtypes.
          3) Converts date and category columns appropriately.
          4) Detects special values in numeric columns (if one value occurs in ≥90% of non-null cells).
          5) Splits numeric columns having special values by creating a corresponding _cat column.
        """
        df = df.copy()

        # (a) Record original dtypes.
        self._record_original_dtypes(df)

        # (b) Auto-assign dtypes for columns not specified.
        self._auto_assign_dtypes(df)

        # (c) Apply the specified dtypes.
        self._apply_user_dtypes(df)

        # (d) Automatically detect special values in numeric columns.
        self._detect_special_values(df)

        # (e) Split numeric columns that have special values.
        self._split_numeric_columns(df)

        # (f) Oversampling the minority
        if oversample == True:
            self._oversample_minority(df)

        return df

    def _record_original_dtypes(self, df: pd.DataFrame):
        for col in df.columns:
            self.original_dtypes[col] = str(df[col].dtype)

    def _auto_assign_dtypes(self, df: pd.DataFrame):
        """
        For columns not specified in user_dtypes, assigns:
          - 'date' if the column is datetime-like.
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
        Converts columns based on assigned dtypes:
          - 'date': uses pd.to_datetime.
          - 'category': converts to category dtype.
          - 'numeric': no conversion.
        """
        for col, dtype_str in self.user_dtypes.items():
            if col not in df.columns:
                continue

            if dtype_str == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype_str == "category":
                df[col] = df[col].astype(
                    pd.CategoricalDtype(
                        categories=df[col].unique().tolist(),
                        ordered=True
                    )
                )

    def _detect_special_values(self, df: pd.DataFrame):
        """
        For each numeric column (as per user_dtypes), if one value occurs in ≥90% of non-null entries,
        automatically add that value as a special value.
        """
        for col in df.columns:
            if self.user_dtypes.get(col, None) == "category":
                series = df[col].dropna()
                freq = series.value_counts(normalize=True)
                max_prop = freq.iloc[0]
                dominant_val = freq.index[0]
                if max_prop >= 0.8:
                    print(f"Categorical Column '{col} is highly imbalanced: {dominant_val} occurs in {max_prop*100:.1f}% of non-null rows.")
                    self.dominant_values[col] = dominant_val

            if self.user_dtypes.get(col, None) != "numeric":
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            freq = series.value_counts(normalize=True)
            if not freq.empty:
                max_prop = freq.iloc[0]
                dominant_val = freq.index[0]
                if max_prop >= 0.8:
                    print(f"[detect_special] Column '{col}' is highly imbalanced: {dominant_val} occurs in {max_prop*100:.1f}% of non-null rows.")
                    self.dominant_values[col] = dominant_val
                    # Merge with any existing user-specified special values.
                    if col in self.user_special_values:
                        if dominant_val not in self.user_special_values[col]:
                            self.user_special_values[col].append(dominant_val)
                    else:
                        self.user_special_values[col] = [dominant_val]

    def _split_numeric_columns(self, df: pd.DataFrame):
        """
        For each numeric column in user_special_values:
          - Create a new numeric column (named base_col_cat) that marks special values using integer markers.
          - For each cell in the base column:
                If NaN -> returns the missing marker (-999999999).
                If the value is in the list of special values -> returns that special value.
                Otherwise -> returns the numeric marker (-777777777).
          - The resulting _cat column is then cast to the same dtype as the original column.
        """
        for col, specials in self.user_special_values.items():
            if col not in df.columns:
                continue

            cat_col = col + "_cat"
            self.split_map[col] = cat_col
            self.detected_specials[col] = specials

            if cat_col in df.columns:
                df.drop(columns=[cat_col], inplace=True)
            base_idx = df.columns.get_loc(col)
            df.insert(base_idx, cat_col, None)

            def cat_mapper(x, specials, normal_marker, missing_marker):
                if pd.isna(x):
                    return float(missing_marker)
                elif x in specials:
                    return float(x)  # retain the special value as-is
                else:
                    return float(normal_marker)

            normal_marker = self.NUMERIC_MARKER
            missing_marker = self.MISSING_MARKER

            df[cat_col] = df[col].apply(
                lambda x: cat_mapper(x, specials, normal_marker, missing_marker)
            ).astype(df[col].dtype)  # Preserve original dtype
    
    def _oversample_minority(self, df: pd.DataFrame):
        for col in self.dominant_values.keys():
            oversample = df[df[col] != self.dominant_values[col]].sample()
            df = concat(df, oversample)

    # =========================================================================
    # POSTPROCESSING
    # =========================================================================
    def postprocess(self, df: pd.DataFrame, rules: Optional[Dict[str, List[Tuple[str, str, Any]]]] = None) -> pd.DataFrame:
        """
        Postprocesses the synthetic DataFrame:
          1) Merges back split columns (base_col, base_col_cat). For rows where the _cat column 
             indicates a missing value, the base column is set to NaN; if it indicates a special value,
             that special value is restored in the base column.
          2) Optionally applies user-provided rules sequentially to filter rows.
          3) Restores the original column order and dtypes.
        """
        df = df.copy()
        
        # (1) Merge split columns.
        df = self._merge_splitted_cols(df)
        
        # (2) Apply business rules if provided.
        if rules is not None:
            df = self.apply_rules(df, rules)
            
        # (3) Restore original column order and dtypes.
        df = df[list(self.original_dtypes.keys())].astype(self.original_dtypes)
        
        return df

    def _merge_splitted_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge back the split columns (base_col and _cat) into original format."""
        for base_col, cat_col in self.split_map.items():
            if cat_col not in df.columns:
                continue

            # Get original dtype before modification.
            original_dtype = self.original_dtypes.get(base_col, np.float64)
            
            # (1) Handle missing values: if _cat equals the missing marker, set base_col to NaN.
            missing_mask = df[cat_col] == self.MISSING_MARKER
            df.loc[missing_mask, base_col] = np.nan
            
            # (2) Apply special values:
            # If _cat is not equal to the numeric or missing marker, restore special value.
            special_mask = ~df[cat_col].isin([self.NUMERIC_MARKER, self.MISSING_MARKER])
            df.loc[special_mask, base_col] = df.loc[special_mask, cat_col].astype(original_dtype)
            
            # (3) Remove auxiliary column.
            df.drop(columns=cat_col, inplace=True)
            
        return df

    def _convert_special_value(self, val: Any, specials: List[Any]) -> Any:
        # Placeholder for additional logic if needed.
        return val

    def apply_rules(self, df: pd.DataFrame, rules: Dict[str, List[Tuple[str, str, Any]]]) -> pd.DataFrame:
        """
        Applies a set of rules to the DataFrame by iteratively dropping rows that do not satisfy each rule.
        
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

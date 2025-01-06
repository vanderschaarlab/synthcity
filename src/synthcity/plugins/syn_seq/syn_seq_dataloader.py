# File: plugins/seq_reg/seq_reg_dataloader.py

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from .syn_seq_encoder import Syn_SeqEncoder


class Syn_SeqDataLoader(DataLoader):
    """
    A DataLoader that applies Syn_Seq-style preprocessing to input data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_order: List[str],
        columns_special_values: Optional[Dict[str, Any]] = None,
        unique_value_threshold: int = 20,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Syn_SeqDataLoader with preprocessing parameters and data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_order (List[str]): Columns to retain and process in specific order.
            columns_special_values (Optional[Dict[str, Any]]): Mapping of columns to special values. Keys are column names, and values are the special values to be replaced.
            unique_value_threshold (int): Threshold to classify columns as numeric or categorical.
            **kwargs (Any): Additional arguments for the DataLoader superclass.
        """
        # Validate target columns
        missing_columns = set(target_order) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        self.target_order = target_order
        self.columns_special_values = columns_special_values or {}
        self.unique_value_threshold = unique_value_threshold

        filtered_data = data[self.target_order]

        super().__init__(
            data_type="Syn_Seq",
            data=filtered_data,
            static_features=list(filtered_data.columns),
            **kwargs,
        )

    def encode(
        self,
        encoders: Optional[Dict[str, Any]] = None,
    ) -> Tuple["DataLoader", Dict]:
        """
        Encode the data using Syn_SeqEncoder with fit/transform pattern.
        """
        if encoders is None:
            encoder = Syn_SeqEncoder(
                columns_special_values=self.columns_special_values,
                target_order=self.target_order,
                unique_value_threshold=self.unique_value_threshold,
            )
            encoder.fit(self.dataframe())
            encoded_data = encoder.transform(self.dataframe())

            return self.decorate(encoded_data), {"syn_seq_encoder": encoder}

        return super().encode(encoders)

    def decode(
        self,
        encoders: Dict[str, Any],
    ) -> "DataLoader":
        """
        Decode the data using stored encoder.
        """
        if "syn_seq_encoder" in encoders:
            encoder = encoders["syn_seq_encoder"]
            if not isinstance(encoder, Syn_SeqEncoder):
                raise TypeError(f"Expected Syn_SeqEncoder, got {type(encoder)}")

            if not hasattr(encoder, "column_order_"):
                raise ValueError(
                    "The encoder has not been fitted. Call fit before using this encoder."
                )

            try:
                decoded_data = encoder.inverse_transform(self.dataframe())
                return self.decorate(decoded_data)
            except Exception as e:
                raise RuntimeError(f"Failed to decode data: {str(e)}")

        return super().decode(encoders)

    def decorate(self, data: pd.DataFrame) -> "Syn_SeqDataLoader":
        """
        Create a new instance of Syn_SeqDataLoader with modified data.
        """
        return Syn_SeqDataLoader(
            data=data,
            target_order=self.target_order,
            columns_special_values=self.columns_special_values,
            unique_value_threshold=self.unique_value_threshold,
        )

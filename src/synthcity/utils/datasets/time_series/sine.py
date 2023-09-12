# stdlib
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd


class SineDataloader:
    """Sine data generation.

    Args:

    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data

    """

    def __init__(
        self,
        no: int = 100,
        seq_len: int = 10,
        temporal_dim: int = 5,
        static_dim: int = 4,
        freq_scale: float = 1,
        as_numpy: bool = False,
        with_missing: bool = False,
    ) -> None:
        self.no = no
        self.seq_len = seq_len
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.freq_scale = freq_scale
        self.as_numpy = as_numpy
        self.with_missing = with_missing

    def load(
        self,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], List, pd.DataFrame]:
        # Initialize the output

        static_data = pd.DataFrame(np.random.rand(self.no, self.static_dim))
        static_data.columns = static_data.columns.astype(str)
        temporal_data = []
        observation_times = []
        outcome = pd.DataFrame(np.random.randint(0, 2, self.no))
        outcome.columns = outcome.columns.astype(str)

        # Generate sine data

        for i in range(self.no):

            # Initialize each time-series
            local = list()

            # For each feature
            if self.with_missing:
                seq_len = np.random.randint(2, self.seq_len, 1)[0]
            else:
                seq_len = self.seq_len

            for k in range(self.temporal_dim):

                # Randomly drawn frequency and phase
                freq = np.random.beta(2, 2)
                phase = np.random.normal()

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [
                    np.sin(self.freq_scale * freq * j + phase) for j in range(seq_len)
                ]

                local.append(temp_data)

            # Align row/column
            # DataFrame with index - time, and columns - temporal features
            local_data = pd.DataFrame(np.transpose(np.asarray(local)))
            local_data.columns = local_data.columns.astype(str)

            # Stack the generated data
            temporal_data.append(local_data)
            observation_times.append(list(range(seq_len)))

        if self.as_numpy:
            return (
                np.asarray(static_data),
                np.asarray(temporal_data),
                np.asarray(observation_times),
                np.asarray(outcome),
            )

        return static_data, temporal_data, observation_times, outcome

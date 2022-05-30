# third party
import numpy as np


class SineDataloader:
    """Sine data generation.

    Args:

    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data

    """

    def __init__(self, no: int, seq_len: int, dim: int, freq_scale: float = 1) -> None:
        self.no = no
        self.seq_len = seq_len
        self.dim = dim
        self.freq_scale = freq_scale

    def load(self) -> np.ndarray:
        # Initialize the output

        data = list()

        # Generate sine data

        for i in range(self.no):

            # Initialize each time-series
            local = list()

            # For each feature
            for k in range(self.dim):

                # Randomly drawn frequency and phase
                freq = np.random.beta(2, 2)
                phase = np.random.normal()

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [
                    np.sin(self.freq_scale * freq * j + phase)
                    for j in range(self.seq_len)
                ]

                local.append(temp_data)

            # Align row/column
            local_data = np.transpose(np.asarray(local))

            # Stack the generated data
            data.append(local_data)

        return np.asarray(data)

# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.core.models.ts_tabular_vae import TimeSeriesTabularVAE
from synthcity.plugins.core.schema import Schema
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_network_config() -> None:
    static, temporal, observation_times, _ = SineDataloader().load()
    net = TimeSeriesTabularVAE(
        static,
        temporal,
        observation_times,
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_nonlin_out_discrete="sigmoid",
        decoder_nonlin_out_continuous="relu",
        decoder_batch_norm=False,
        decoder_dropout=0,
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        # Training
        weight_decay=1e-3,
        n_iter=1001,
        lr=1e-3,
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
        encoder_max_clusters=11,
    )

    assert net.model is not None
    assert net.encoder is not None
    assert net.encoder.max_clusters == 11
    assert net.model.batch_size == 64
    assert net.model.n_iter == 1001
    assert net.model.random_state == 77


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_vae_generation(source: Any) -> None:
    static, temporal, observation_times, _ = source().load()

    model = TimeSeriesTabularVAE(
        static,
        temporal,
        observation_times,
        n_iter=10,
    )
    model.fit(static, temporal, observation_times)

    static_gen, temporal_gen, observation_times_gen = model.generate(10)

    assert static_gen.shape == (10, static.shape[1])
    assert len(observation_times_gen) == len(temporal_gen)
    assert np.asarray(temporal_gen).shape == (
        10,
        temporal[0].shape[0],
        temporal[0].shape[1],
    )


@pytest.mark.slow
@pytest.mark.parametrize("source", [GoogleStocksDataloader])
def test_ts_vae_generation_schema(source: Any) -> None:
    static, temporal, observation_times, _ = source().load()

    model = TimeSeriesTabularVAE(
        static,
        temporal,
        observation_times,
        n_iter=10,
    )
    model.fit(static, temporal, observation_times)

    static_gen, temporal_gen, observation_times_gen = model.generate(1000)

    reference_data = TimeSeriesDataLoader(
        temporal_data=temporal,
        observation_times=observation_times,
        static_data=static,
    )
    reference_schema = Schema(data=reference_data)

    gen_data = TimeSeriesDataLoader(
        temporal_data=temporal_gen,
        observation_times=observation_times_gen,
        static_data=static_gen,
    )

    seq_df = gen_data.dataframe()
    assert reference_schema.as_constraints().filter(seq_df).sum() > 0

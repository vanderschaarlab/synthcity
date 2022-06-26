# stdlib
from typing import Any

# third party
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.core.models.ts_model import modes
from synthcity.plugins.core.models.ts_vae import TimeSeriesAutoEncoder
from synthcity.plugins.core.schema import Schema
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_network_config() -> None:
    net = TimeSeriesAutoEncoder(
        n_static_units=10,
        n_static_units_embedding=2,
        n_temporal_units=11,
        n_temporal_window=2,
        n_temporal_units_embedding=3,
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_static_nonlin_out=[("sigmoid", 10)],
        decoder_temporal_nonlin_out=[("sigmoid", 22)],
        n_iter=1001,
        decoder_batch_norm=False,
        decoder_dropout=0,
        decoder_mode="LSTM",
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        encoder_mode="LSTM",
        # Training
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
    )

    assert net.decoder is not None
    assert net.encoder is not None
    assert net.batch_size == 64
    assert net.n_iter == 1001
    assert net.random_state == 77


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("n_iter", [10])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
@pytest.mark.parametrize("mode", modes)
def test_basic_network(
    nonlin: str,
    n_iter: int,
    dropout: float,
    batch_norm: bool,
    lr: float,
    mode: str,
) -> None:
    net = TimeSeriesAutoEncoder(
        n_static_units=10,
        n_static_units_embedding=2,
        n_temporal_units=11,
        n_temporal_window=2,
        n_temporal_units_embedding=2,
        n_iter=n_iter,
        decoder_dropout=dropout,
        encoder_dropout=dropout,
        decoder_nonlin=nonlin,
        encoder_nonlin=nonlin,
        decoder_batch_norm=batch_norm,
        encoder_batch_norm=batch_norm,
        decoder_n_layers_hidden=2,
        encoder_n_layers_hidden=2,
        lr=lr,
        encoder_mode=mode,
        decoder_mode=mode,
    )

    assert net.n_iter == n_iter
    assert net.lr == lr
    assert net.decoder.mode == mode
    assert net.encoder.mode == mode


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_ode_generation(source: Any) -> None:
    static, temporal, temporal_horizons, _ = source(as_numpy=True).load()

    model = TimeSeriesAutoEncoder(
        n_static_units=static.shape[-1],
        n_static_units_embedding=static.shape[-1],
        n_temporal_units=temporal.shape[-1],
        n_temporal_window=temporal.shape[-2],
        n_temporal_units_embedding=temporal.shape[-1],
        n_iter=20,
    )
    model.fit(static, temporal, temporal_horizons)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(10)

    assert static_gen.shape == (10, static.shape[1])
    assert temporal_gen.shape == (10, temporal.shape[1], temporal.shape[2])
    assert temporal_horizons_gen.shape == (10, temporal.shape[1])


@pytest.mark.slow
@pytest.mark.parametrize("source", [GoogleStocksDataloader])
def test_ts_ode_generation_schema(source: Any) -> None:
    static, temporal, temporal_horizons, _ = source().load()

    model = TimeSeriesAutoEncoder(
        n_static_units=static.shape[-1],
        n_static_units_embedding=static.shape[-1],
        n_temporal_units=temporal[0].shape[-1],
        n_temporal_window=temporal[0].shape[0],
        n_temporal_units_embedding=temporal[0].shape[-1],
        n_iter=100,
    )
    model.fit(static, temporal, temporal_horizons)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(1000)

    reference_data = TimeSeriesDataLoader(
        temporal_data=temporal,
        temporal_horizons=temporal_horizons,
        static_data=static,
    )
    reference_schema = Schema(data=reference_data)

    temporal_list = []
    for item in temporal_gen:
        temporal_list.append(pd.DataFrame(item, columns=temporal[0].columns))
    gen_data = TimeSeriesDataLoader(
        temporal_data=temporal_list,
        temporal_horizons=temporal_horizons_gen.tolist(),
        static_data=pd.DataFrame(static_gen, columns=static.columns),
    )

    seq_df = gen_data.dataframe()

    assert reference_schema.as_constraints().filter(seq_df).sum() > 0

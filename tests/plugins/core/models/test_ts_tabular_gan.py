# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.core.models.ts_tabular_gan import TimeSeriesTabularGAN
from synthcity.plugins.core.schema import Schema
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_network_config() -> None:
    static, temporal, temporal_horizons, _ = SineDataloader().load()
    net = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        # Generator
        generator_n_layers_hidden=2,
        generator_n_units_hidden=100,
        generator_nonlin="elu",
        generator_nonlin_out_discrete="sigmoid",
        generator_nonlin_out_continuous="relu",
        generator_n_iter=1001,
        generator_batch_norm=False,
        generator_dropout=0,
        generator_lr=1e-3,
        generator_weight_decay=1e-3,
        # Discriminator
        discriminator_n_layers_hidden=3,
        discriminator_n_units_hidden=100,
        discriminator_nonlin="elu",
        discriminator_n_iter=1002,
        discriminator_batch_norm=False,
        discriminator_dropout=0,
        discriminator_lr=1e-3,
        discriminator_weight_decay=1e-3,
        # Training
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
        encoder_max_clusters=11,
        gamma_penalty=2,
        moments_penalty=2,
        embedding_penalty=2,
    )

    assert net.model is not None
    assert net.encoder is not None
    assert net.encoder.max_clusters == 11
    assert net.model.batch_size == 64
    assert net.model.generator_n_iter == 1001
    assert net.model.discriminator_n_iter == 1002
    assert net.model.random_state == 77
    assert net.model.gamma_penalty == 2
    assert net.model.moments_penalty == 2
    assert net.model.embedding_penalty == 2


@pytest.mark.slow
@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_gan_generation(source: Any) -> None:
    static, temporal, temporal_horizons, _ = source().load()

    model = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        generator_n_iter=10,
    )
    model.fit(static, temporal, temporal_horizons)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(10)

    assert static_gen.shape == (10, static.shape[1])
    assert len(temporal_horizons_gen) == len(temporal_gen)
    assert np.asarray(temporal_gen).shape == (
        10,
        temporal[0].shape[0],
        temporal[0].shape[1],
    )


@pytest.mark.slow
@pytest.mark.parametrize("source", [GoogleStocksDataloader])
def test_ts_gan_generation_schema(source: Any) -> None:
    static, temporal, temporal_horizons, _ = source().load()

    model = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        generator_n_iter=10,
    )
    model.fit(static, temporal, temporal_horizons)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(1000)

    reference_data = TimeSeriesDataLoader(
        temporal_data=temporal,
        temporal_horizons=temporal_horizons,
        static_data=static,
    )
    reference_schema = Schema(data=reference_data)

    gen_data = TimeSeriesDataLoader(
        temporal_data=temporal_gen,
        temporal_horizons=temporal_horizons_gen,
        static_data=static_gen,
    )

    seq_df = gen_data.dataframe()
    assert reference_schema.as_constraints().filter(seq_df).sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_tabular_gan_conditional(source: Any) -> None:
    static, temporal, temporal_horizons, outcome = source().load()

    model = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        n_units_conditional=1,
        generator_n_iter=10,
    )
    model.fit(static, temporal, temporal_horizons, cond=outcome)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(10)
    assert static_gen.shape == (10, static.shape[1])
    assert len(temporal_horizons_gen) == len(temporal_gen)
    assert np.asarray(temporal_gen).shape == (
        10,
        temporal[0].shape[0],
        temporal[0].shape[1],
    )

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(
        5, np.ones([5, *outcome.shape[1:]])
    )
    assert static_gen.shape == (5, static.shape[1])
    assert len(temporal_horizons_gen) == len(temporal_gen)
    assert np.asarray(temporal_gen).shape == (
        5,
        temporal[0].shape[0],
        temporal[0].shape[1],
    )


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_tabular_gan_conditional_static_data(source: Any) -> None:
    static, temporal, temporal_horizons, outcome = source().load()

    model = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        generator_n_iter=10,
    )
    model.fit(static, temporal, temporal_horizons)

    cnt = 10
    static_seed = static.sample(cnt)
    static_gen, temporal_gen, temporal_horizons_gen = model.generate(
        cnt, static_data=static_seed
    )
    assert np.isclose(np.asarray(static_gen), np.asarray(static_seed)).all()
    assert np.asarray(temporal_gen).shape == (
        cnt,
        len(temporal[0]),
        temporal[0].shape[1],
    )
    assert np.asarray(temporal_horizons_gen).shape == (cnt, len(temporal[0]))


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_tabular_gan_conditional_temporal_horizons(source: Any) -> None:
    static, temporal, temporal_horizons, outcome = source().load()

    model = TimeSeriesTabularGAN(
        static,
        temporal,
        temporal_horizons,
        generator_n_iter=10,
    )
    model.fit(static, temporal, temporal_horizons)

    cnt = 10
    temporal_horizons = np.asarray(temporal_horizons)
    horizon_ids = np.random.choice(len(temporal_horizons), cnt, replace=False)

    static_gen, temporal_gen, temporal_horizons_gen = model.generate(
        cnt, temporal_horizons=temporal_horizons[horizon_ids].tolist()
    )
    assert np.isclose(
        np.asarray(temporal_horizons_gen), np.asarray(temporal_horizons[horizon_ids])
    ).all()
    assert np.asarray(temporal_gen).shape == (
        cnt,
        len(temporal[0]),
        temporal[0].shape[1],
    )
    assert np.asarray(temporal_horizons_gen).shape == (cnt, len(temporal[0]))

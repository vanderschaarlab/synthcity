# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# synthcity absolute
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins.core.models.ts_gan import TimeSeriesGAN
from synthcity.plugins.core.models.ts_model import modes
from synthcity.plugins.core.schema import Schema
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.sine import SineDataloader


def test_network_config() -> None:
    net = TimeSeriesGAN(
        n_static_units=10,
        n_static_units_latent=2,
        n_temporal_units=11,
        n_temporal_window=2,
        n_temporal_units_latent=3,
        # Generator
        generator_n_layers_hidden=2,
        generator_n_units_hidden=100,
        generator_nonlin="elu",
        generator_static_nonlin_out=[("sigmoid", 10)],
        generator_temporal_nonlin_out=[("sigmoid", 22)],
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
        gamma_penalty=2,
        moments_penalty=2,
        embedding_penalty=2,
    )

    assert net.static_generator is not None
    assert net.temporal_generator is not None
    assert net.discriminator is not None
    assert net.batch_size == 64
    assert net.generator_n_iter == 1001
    assert net.discriminator_n_iter == 1002
    assert net.random_state == 77
    assert net.gamma_penalty == 2
    assert net.moments_penalty == 2
    assert net.embedding_penalty == 2


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
    net = TimeSeriesGAN(
        n_static_units=10,
        n_static_units_latent=2,
        n_temporal_units=11,
        n_temporal_window=2,
        n_temporal_units_latent=2,
        generator_n_iter=n_iter,
        discriminator_n_iter=n_iter,
        generator_dropout=dropout,
        discriminator_dropout=dropout,
        generator_nonlin=nonlin,
        discriminator_nonlin=nonlin,
        generator_batch_norm=batch_norm,
        discriminator_batch_norm=batch_norm,
        generator_n_layers_hidden=2,
        discriminator_n_layers_hidden=2,
        generator_lr=lr,
        discriminator_lr=lr,
        mode=mode,
    )

    assert net.generator_n_iter == n_iter
    assert net.discriminator_n_iter == n_iter
    assert net.static_generator.lr == lr
    assert net.temporal_generator.lr == lr
    assert net.temporal_generator.mode == mode
    assert net.discriminator.lr == lr
    assert net.discriminator.mode == mode


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_gan_generation(source: Any) -> None:
    static, temporal, observation_times, _ = source(as_numpy=True).load()

    model = TimeSeriesGAN(
        n_static_units=static.shape[-1],
        n_static_units_latent=static.shape[-1],
        n_temporal_units=temporal.shape[-1],
        n_temporal_window=temporal.shape[-2],
        n_temporal_units_latent=temporal.shape[-1],
        generator_n_iter=10,
    )
    model.fit(static, temporal, observation_times)

    static_gen, temporal_gen, observation_times_gen = model.generate(10)

    assert static_gen.shape == (10, static.shape[1])
    assert temporal_gen.shape == (10, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (10, temporal.shape[1])


@pytest.mark.slow
@pytest.mark.parametrize("source", [GoogleStocksDataloader])
def test_ts_gan_generation_schema(source: Any) -> None:
    static, temporal, observation_times, _ = source().load()

    model = TimeSeriesGAN(
        n_static_units=static.shape[-1],
        n_static_units_latent=static.shape[-1],
        n_temporal_units=temporal[0].shape[-1],
        n_temporal_window=temporal[0].shape[0],
        n_temporal_units_latent=temporal[0].shape[-1],
        generator_n_iter=100,
    )
    model.fit(static, temporal, observation_times)

    static_gen, temporal_gen, observation_times_gen = model.generate(1000)

    reference_data = TimeSeriesDataLoader(
        temporal_data=temporal,
        observation_times=observation_times,
        static_data=static,
    )
    reference_schema = Schema(data=reference_data)

    temporal_list = []
    for item in temporal_gen:
        temporal_list.append(pd.DataFrame(item, columns=temporal[0].columns))
    gen_data = TimeSeriesDataLoader(
        temporal_data=temporal_list,
        observation_times=observation_times_gen.tolist(),
        static_data=pd.DataFrame(static_gen, columns=static.columns),
    )

    seq_df = gen_data.dataframe()

    assert reference_schema.as_constraints().filter(seq_df).sum() > 0


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_gan_conditional(source: Any) -> None:
    static, temporal, observation_times, outcome = source(as_numpy=True).load()

    model = TimeSeriesGAN(
        n_static_units=static.shape[-1],
        n_static_units_latent=static.shape[-1],
        n_temporal_units=temporal.shape[-1],
        n_temporal_window=temporal.shape[-2],
        n_temporal_units_latent=temporal.shape[-1],
        n_units_conditional=1,
        generator_n_iter=10,
    )
    model.fit(static, temporal, observation_times, cond=outcome)

    static_gen, temporal_gen, observation_times_gen = model.generate(10)
    assert static_gen.shape == (10, static.shape[1])
    assert temporal_gen.shape == (10, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (10, temporal.shape[1])

    static_gen, temporal_gen, observation_times_gen = model.generate(
        5, np.ones([5, *outcome.shape[1:]])
    )
    assert static_gen.shape == (5, static.shape[1])
    assert temporal_gen.shape == (5, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (5, temporal.shape[1])


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_gan_conditional_static_data(source: Any) -> None:
    static, temporal, observation_times, outcome = source(as_numpy=True).load()

    model = TimeSeriesGAN(
        n_static_units=static.shape[-1],
        n_static_units_latent=static.shape[-1],
        n_temporal_units=temporal.shape[-1],
        n_temporal_window=temporal.shape[-2],
        n_temporal_units_latent=temporal.shape[-1],
        n_units_conditional=1,
        generator_n_iter=10,
    )
    model.fit(static, temporal, observation_times, cond=outcome)

    cnt = 10
    static_ids = np.random.choice(len(static), cnt, replace=False)
    static_gen, temporal_gen, observation_times_gen = model.generate(
        cnt, static_data=static[static_ids]
    )
    assert np.isclose(np.asarray(static_gen), np.asarray(static[static_ids])).all()
    assert temporal_gen.shape == (cnt, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (cnt, temporal.shape[1])

    cnt = 5
    static_ids = np.random.choice(len(static), cnt, replace=False)
    static_gen, temporal_gen, observation_times_gen = model.generate(
        cnt, cond=np.ones([cnt, *outcome.shape[1:]]), static_data=static[static_ids]
    )
    assert np.isclose(np.asarray(static_gen), np.asarray(static[static_ids])).all()
    assert temporal_gen.shape == (cnt, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (cnt, temporal.shape[1])


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_gan_conditional_observation_times(source: Any) -> None:
    static, temporal, observation_times, outcome = source(as_numpy=True).load()

    model = TimeSeriesGAN(
        n_static_units=static.shape[-1],
        n_static_units_latent=static.shape[-1],
        n_temporal_units=temporal.shape[-1],
        n_temporal_window=temporal.shape[-2],
        n_temporal_units_latent=temporal.shape[-1],
        n_units_conditional=1,
        generator_n_iter=10,
    )
    model.fit(static, temporal, observation_times, cond=outcome)

    cnt = 10
    observation_times = np.asarray(observation_times)
    horizon_ids = np.random.choice(len(observation_times), cnt, replace=False)
    static_gen, temporal_gen, observation_times_gen = model.generate(
        cnt, observation_times=observation_times[horizon_ids]
    )
    assert np.isclose(
        np.asarray(observation_times_gen), np.asarray(observation_times[horizon_ids])
    ).all()
    assert temporal_gen.shape == (cnt, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (cnt, temporal.shape[1])

    cnt = 5
    horizon_ids = np.random.choice(len(observation_times), cnt, replace=False)
    static_gen, temporal_gen, observation_times_gen = model.generate(
        cnt,
        cond=np.ones([cnt, *outcome.shape[1:]]),
        observation_times=observation_times[horizon_ids],
    )
    assert np.isclose(
        np.asarray(observation_times_gen), np.asarray(observation_times[horizon_ids])
    ).all()
    assert temporal_gen.shape == (cnt, temporal.shape[1], temporal.shape[2])
    assert observation_times_gen.shape == (cnt, temporal.shape[1])

# third party
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
)


def test_generic_dataloader() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X["target"] = y

    loader = GenericDataLoader(X)

    assert loader.raw().shape == X.shape
    assert loader.type() == "generic"
    assert loader.shape == X.shape
    assert list(loader.columns) == list(X.columns)
    assert (loader.numpy() == X.values).all()
    assert len(loader) == len(X)

    assert loader.info()["data_type"] == "generic"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(X.columns)


def test_survival_dataloader() -> None:
    df = load_rossi()

    target_column = "arrest"
    time_to_event_column = "week"

    X = df.drop(columns=[target_column, time_to_event_column])
    T = df[time_to_event_column]
    E = df[target_column]

    loader = SurvivalAnalysisDataLoader(
        df,
        time_to_event_column=time_to_event_column,
        target_column=target_column,
        time_horizons=[20],
    )

    assert loader.raw().shape == df.shape
    assert loader.preprocessed()[0].shape == X.shape
    assert loader.preprocessed()[1].shape == T.shape
    assert loader.preprocessed()[2].shape == E.shape

    assert loader.type() == "survival_analysis"
    assert loader.shape == df.shape
    assert sorted(list(loader.dataframe().columns)) == sorted(list(df.columns))
    assert (loader.numpy() == df.values).all()
    assert len(loader) == len(X)

    assert loader.info()["data_type"] == "survival_analysis"
    assert loader.info()["len"] == len(X)
    assert loader.info()["static_features"] == list(df.columns)


def test_time_series_dataloader() -> None:
    pass

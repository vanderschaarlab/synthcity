import pandas as pd
import math
from matplotlib import pyplot as plt
import os

# make sure that discrete features have max. k features, set in tabular_vae.py
# or change code of tabular_vae.py such that it takes discrete features as input


def preprocess(X: pd.DataFrame, y: pd.DataFrame, config: dict):
    # drop features
    X = X.drop(config["drop"], axis=1)

    # cast numericals
    X[config["numerical"]] = X[config["numerical"]].astype(float)

    y = y.squeeze()
    # dataset specific preprocessing
    if config["name"] == "adult":
        X.loc[
            ~X["native-country"].isin(["United-States", "Mexico"]), "native-country"
        ] = "Other"
        y = y.replace({"<=50K": 0.0, "<=50K.": 0.0, ">50K": 1.0, ">50K.": 1.0})
    elif config["name"] == "credit":
        X = X.rename({"X2": "sex", "X5": "age"}, axis=1)
    elif config["name"] == "heart":
        y = y.apply(lambda x: 1 if x > 0 else 0)
    elif config["name"] == "student":
        # target is whether student fails the class (grade below 10 out of 20)
        y = y.G3
        y = y < 10
    elif config["name"] == "diabetes":
        for col in config["discrete"]:
            # fill missing categories (no missing numericals exist in the dataset)
            X[col] = X[col].fillna("missing")
            # frequency encode high cardinality features
            if X[col].nunique() > 15:
                X[col] = frequency_encode(X[col])

    # add y as target to df
    y.name = "target"
    df = pd.concat([X, y], axis=1)
    return df


def frequency_encode(series):
    freq_map = series.value_counts().to_dict()
    return series.map(freq_map)


def plot_df(df: pd.DataFrame):
    # Determine the number of features
    num_features = len(df.columns)

    # Determine grid size (rows and columns)
    grid_size = math.ceil(math.sqrt(num_features))

    # Create a figure and axes with a grid layout
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each feature
    for i, col in enumerate(df.columns):
        if df[col].dtype == "object":  # Discrete feature
            df[col].value_counts().plot(kind="bar", ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        else:  # Numerical feature
            df[col].plot(kind="hist", ax=axes[i], bins=10)
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def clear_dir(directory):
    # remove all files from specified directory
    for item in os.listdir(directory):
        filepath = os.path.join(directory, item)
        if os.path.isfile(filepath):
            os.remove(filepath)

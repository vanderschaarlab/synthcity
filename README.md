<h2 align="center">
  <img src="https://github.com/vanderschaarlab/synthcity/raw/main/docs/logo.png" height="150px">

  synthcity
</h2>

<h4 align="center">
    A library for generating and evaluating synthetic tabular data.
</h4>


<div align="center">

[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vr2PJswgfFYBkJCm3hhVkuH-9dXnHeYV?usp=sharing)
[![Tests Quick Python](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_pr.yml/badge.svg)](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_pr.yml)
[![Tests Full Python](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_full.yml/badge.svg)](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_full.yml)
[![Tutorials](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/vanderschaarlab/synthcity/actions/workflows/test_tutorials.yml)
[![Documentation Status](https://readthedocs.org/projects/synthcity/badge/?version=latest)](https://synthcity.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2301.07573-b31b1b.svg)](https://arxiv.org/abs/2301.07573)

[![](https://pepy.tech/badge/synthcity)](https://pypi.org/project/synthcity/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/synthcity/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1pzy8z7ti-zVsUPHAKTgCd1UoY8XtTEw)

![image](https://github.com/vanderschaarlab/synthcity/raw/main/docs/arch.png "Synthcity")

</div>


## Features:
- :key: Easy to extend pluginable architecture.
- :cyclone: Several evaluation metrics for correctness and privacy.
- :fire: Several reference models, by type:
    - General purpose: GAN-based (AdsGAN, CTGAN, PATEGAN, DP-GAN),VAE-based(TVAE, RTVAE), Normalizing flows, Bayesian Networks(PrivBayes, BN), Random Forrest (arfpy), LLM-based (GReaT).
    - Time Series & Time-Series Survival generators: TimeGAN, FourierFlows, TimeVAE.
    - Static Survival Analysis: SurvivalGAN, SurVAE.
    - Privacy-focused: DECAF, DP-GAN, AdsGAN, PATEGAN, PrivBayes.
    - Domain adaptation: RadialGAN.
    - Images: Image ConditionalGAN, Image AdsGAN.
- :book: [Read the docs !](https://synthcity.readthedocs.io/)
- :airplane: [Checkout the tutorials!](https://github.com/vanderschaarlab/synthcity#-tutorials)

*Please note: synthcity does not handle missing data and so these values must be imputed first [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) can be used to do this.*

## :rocket: Installation

The library can be installed from PyPI using
```bash
$ pip install synthcity
```
or from source, using
```bash
$ pip install .
```
Other library extensions:
 * Install the library with unit-testing support
```bash
 pip install synthcity[testing]
```
 * Install the library with GOGGLE support
```bash
 pip install synthcity[goggle]
```
 * Install the library with ALL the extensions
```bash
 pip install synthcity[all]
```


## :boom: Sample Usage

### Generic data
* List the available general-purpose generators

```python
from synthcity.plugins import Plugins

Plugins(categories=["generic", "privacy"]).list()
```

* Load and train a tabular generator

```python
from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

syn_model = Plugins().get("adsgan")

syn_model.fit(X)
```

* Generate new synthetic tabular data

```python
syn_model.generate(count = 10)
```

* Benchmark the quality of the plugins

```python
# third party
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

loader = GenericDataLoader(X, target_column="target", sensitive_columns=["sex"])

score = Benchmarks.evaluate(
    [
        (f"example_{model}", model, {})  # testname, plugin name, plugin args
        for model in ["adsgan", "ctgan", "tvae"]
    ],
    loader,
    synthetic_size=1000,
    metrics={"performance": ["linear_model"]},
    repeats=3,
)
Benchmarks.print(score)
```

### Static Survival analysis

* List the available generators dedicated to survival analysis

```python
from synthcity.plugins import Plugins

Plugins(categories=["generic", "privacy", "survival_analysis"]).list()
```

* Generate new data

```python
from lifelines.datasets import load_rossi
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins

X = load_rossi()
data = SurvivalAnalysisDataLoader(
    X,
    target_column="arrest",
    time_to_event_column="week",
)

syn_model = Plugins().get("survival_gan")

syn_model.fit(data)

syn_model.generate(count=10)
```

### Time series

* List the available generators

```python
from synthcity.plugins import Plugins

Plugins(categories=["generic", "privacy", "time_series"]).list()
```

* Generate new data

```python
# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader

static_data, temporal_data, horizons, outcome = GoogleStocksDataloader().load()
data = TimeSeriesDataLoader(
    temporal_data=temporal_data,
    observation_times=horizons,
    static_data=static_data,
    outcome=outcome,
)

syn_model = Plugins().get("timegan")

syn_model.fit(data)

syn_model.generate(count=10)
```
### Images

__Note__ : The architectures used for generators are not state-of-the-art. For other architectures, consider extending the `suggest_image_generator_discriminator_arch` method from the `convnet.py` module.

* List the available generators

```python
from synthcity.plugins import Plugins

Plugins(categories=["images"]).list()
```

* Generate new data
```python
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import ImageDataLoader
from torchvision import datasets


dataset = datasets.MNIST(".", download=True)
loader = ImageDataLoader(dataset).sample(100)

syn_model = Plugins().get("image_cgan")

syn_model.fit(loader)

syn_img, syn_labels = syn_model.generate(count=10).unpack().numpy()

print(syn_img.shape)

```

### Serialization

* Using save/load methods

```python
from synthcity.utils.serialization import save, load
from synthcity.plugins import Plugins

syn_model = Plugins().get("adsgan")

buff = save(syn_model)
reloaded = load(buff)

assert syn_model.name() == reloaded.name()
```

* Saving and loading models from disk

```python
from sklearn.datasets import load_diabetes
from synthcity.utils.serialization import save_to_file, load_from_file
from synthcity.plugins import Plugins

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

syn_model = Plugins().get("adsgan", n_iter=10)

syn_model.fit(X)

save_to_file('./adsgan_10_epochs.pkl', syn_model)
reloaded = load_from_file('./adsgan_10_epochs.pkl')

assert syn_model.name() == reloaded.name()
```

* Using the Serializable interface

```python
from synthcity.plugins import Plugins

syn_model = Plugins().get("adsgan")

buff = syn_model.save()
reloaded = Plugins().load(buff)

assert syn_model.name() == reloaded.name()
```

## 📓 Tutorials

 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial0_basic_examples.ipynb) [ Tutorial 0: Getting started with tabular data](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial0_basic_examples.ipynb)
  - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial1_add_a_new_plugin.ipynb) [ Tutorial 1: Writing a new plugin](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial1_add_a_new_plugin.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial2_benchmarks.ipynb) [ Tutorial 2: Benchmarking models](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial2_benchmarks.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial3_survival_analysis.ipynb) [ Tutorial 3: Generating Survival Analysis data](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial3_survival_analysis.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial4_time_series.ipynb) [ Tutorial 4: Generating Time Series](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial4_time_series.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial5_differential_privacy.ipynb) [ Tutorial 5: Generating Data with Differential Privacy Guarantees](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial5_differential_privacy.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial6_time_series_data_preparation.ipynb) [ Tutorial 6: Practice using Custom Time series data](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial6_time_series_data_preparation.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial7_image_generation_using_mednist.ipynb) [ Tutorial 7: Image generation](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial7_image_generation_using_mednist.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial8_hyperparameter_optimization.ipynb) [ Tutorial 8: hyperparameter optimization](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial8_hyperparameter_optimization.ipynb)
   - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial9_dealing_with_missing_data.ipynb) [ Tutorial 9: dealing with missing data](https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial9_dealing_with_missing_data.ipynb)

## 🔑 Methods

### Bayesian methods
| Method | Description | Reference |
|--- | --- | --- |
|**bayesian_network**| The method represents a set of random variables and their conditional dependencies via a directed acyclic graph (DAG), and uses it to sample new data points| [pgmpy](https://pgmpy.org/)|
|**privbayes**|  A differentially private method for releasing high-dimensional data. | [PrivBayes: Private Data Release via Bayesian Networks](https://dl.acm.org/doi/10.1145/3134428)|

### Generative adversarial networks(GANs)

| Method | Description | Reference |
|--- | --- | --- |
|**adsgan**| A conditional GAN framework that generates synthetic data while minimize patient identifiability that is defined based on the probability of re-identification given the combination of all data on any individual patient|  [Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN)](https://pubmed.ncbi.nlm.nih.gov/32167919/) |
|**pategan**| The methos uses the Private Aggregation of Teacher Ensembles (PATE) framework and applies it to GANs, allowing to tightly bound the influence of any individual sample on the model, resulting in tight differential privacy guarantees and thus an improved performance over models with the same guarantees.|  [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees](https://openreview.net/forum?id=S1zk9iRqF7) |
|**ctgan**| A conditional generative adversarial network which can handle tabular data.| [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)|

### Variational autoencoders(VAE)
| Method | Description | Reference |
|--- | --- | --- |
|**tvae**| A conditional VAE network which can handle tabular data.|  [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) |
|**rtvae**| A robust variational autoencoder with β divergence for tabular data (RTVAE) with mixed categorical and continuous features.|  [Robust Variational Autoencoder for Tabular Data with β Divergence](https://arxiv.org/abs/2006.08204) |


### Normalizing Flows

| Method | Description | Reference |
|--- | --- | --- |
|**nflow**| Normalizing Flows are generative models which produce tractable distributions where both sampling and density evaluation can be efficient and exact.| [Neural Spline Flows](https://arxiv.org/abs/1906.04032) |

### Graph neural networks

| Method | Description | Reference |
|--- | --- | --- |
|**goggle**| GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure | [Paper](https://openreview.net/forum?id=fPVRcJqspu) |


### Diffusion models

| Method | Description | Reference |
|--- | --- | --- |
|**ddpm**| TabDDPM: Modelling Tabular Data with Diffusion Models. | [Paper](https://arxiv.org/abs/2209.15421) |

### Random Forest models

| Method | Description | Reference |
|--- | --- | --- |
|**arfpy**| Adversarial Random Forests for Density Estimation and Generative Modeling | [Paper](https://proceedings.mlr.press/v206/watson23a.html) |

### LLM-based models

| Method | Description | Reference |
|--- | --- | --- |
|**GReaT**| Language Models are Realistic Tabular Data Generators | [Paper](https://openreview.net/forum?id=cEygmQNOeI) |

### Static Survival analysis methods

| Method | Description | Reference |
|--- | --- | --- |
|**survival_gan** | SurvivalGAN is a generative model that can handle survival data by addressing the imbalance in the censoring and time horizons, using a dedicated mechanism for approximating time to event/censoring from the input and survival function. | --- |
|**survival_ctgan** | SurvivalGAN version using CTGAN | --- |
|**survae** | SurvivalGAN version using VAE | --- |
|**survival_nflow** | SurvivalGAN version using normalizing flows | --- |

### Time-Series and Time-Series Survival Analysis methods

| Method | Description | Reference |
|--- | --- | --- |
| **timegan** | TimeGAN is a framework for generating realistic time-series data that combines the flexibility of the unsupervised paradigm with the control afforded by supervised training. Through a learned embedding space jointly optimized with both supervised and adversarial objectives, the network adheres to the dynamics of the training data during sampling.  | [Time-series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) |
| **fflows** |  FFlows is an explicit likelihood model based on a novel class of normalizing flows that view time-series data in the frequency-domain rather than the time-domain. The method uses a discrete Fourier transform (DFT) to convert variable-length time-series with arbitrary sampling periods into fixed-length spectral representations, then applies a (data-dependent) spectral filter to the frequency-transformed time-series.  | [Generative Time-series Modeling with Fourier Flows](https://openreview.net/forum?id=PpshD0AXfA) |


### Privacy & Fairness

| Method | Description | Reference |
|--- | --- | --- |
|**decaf** | Machine learning models have been criticized for reflecting unfair biases in the training data. Instead of solving this by introducing fair learning algorithms directly, DEACF focuses on generating fair synthetic data, such that any downstream learner is fair. Generating fair synthetic data from unfair data - while remaining truthful to the underlying data-generating process (DGP) - is non-trivial. DECAF is a GAN-based fair synthetic data generator for tabular data. With DECAF, we embed the DGP explicitly as a structural causal model in the input layers of the generator, allowing each variable to be reconstructed conditioned on its causal parents. This procedure enables inference time debiasing, where biased edges can be strategically removed to satisfy user-defined fairness requirements.   | [DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks](https://arxiv.org/abs/2110.12884) |
|**privbayes**|  A differentially private method for releasing high-dimensional data. | [PrivBayes: Private Data Release via Bayesian Networks](https://dl.acm.org/doi/10.1145/3134428)|
|**dpgan** | Differentially Private GAN | [Differentially Private Generative Adversarial Network](https://arxiv.org/abs/1802.06739) |
|**adsgan**| A conditional GAN framework that generates synthetic data while minimize patient identifiability that is defined based on the probability of re-identification given the combination of all data on any individual patient|  [Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN)](https://pubmed.ncbi.nlm.nih.gov/32167919/) |
|**pategan**| The methos uses the Private Aggregation of Teacher Ensembles (PATE) framework and applies it to GANs, allowing to tightly bound the influence of any individual sample on the model, resulting in tight differential privacy guarantees and thus an improved performance over models with the same guarantees.|  [PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees](https://openreview.net/forum?id=S1zk9iRqF7) |


### Domain adaptation

| Method | Description | Reference |
|--- | --- | --- |
|**radialgan** | Training complex machine learning models for prediction often requires a large amount of data that is not always readily available. Leveraging these external datasets from related but different sources is, therefore, an essential task if good predictive models are to be built for deployment in settings where data can be rare. RadialGAN is an approach to the problem in which multiple GAN architectures are used to learn to translate from one dataset to another, thereby allowing to augment the target dataset effectively and learning better predictive models than just the target dataset. | [RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks](https://arxiv.org/abs/1802.06403) |

### Images

| Method | Description | Reference |
|--- | --- | --- |
|**image_cgan**| Conditional GAN for generating images|  --- |
|**image_adsgan**| The AdsGAN method adapted for image generation|  --- |


### Debug methods

| Method | Description | Reference |
|--- | --- | --- |
|**marginal_distributions**| A differentially private method that samples from the marginal distributions of the training set|  --- |
|**uniform_sampler**| A differentially private method that uniformly samples from the [min, max] ranges of each column.|  --- |
|**dummy_sampler**| Resample data points from the training set|  --- |



## :zap: Evaluation metrics
The following table contains the available evaluation metrics:

- __Sanity checks__

| Metric | Description| Values |
|--- | --- | --- |
|**data_mismatch**| Average number of columns with datatype(object, real, int) mismatch between the real and synthetic data|0: no datatype mismatch. <br/>1: complete data type mismatch between the datasets.|
|**common_rows_proportion**| The proportion of rows in the real dataset leaked in the synthetic dataset.| 0: there are no common rows between the real and synthetic datasets. <br/> 1: all the rows in the real dataset are leaked in the synthetic dataset. |
|**nearest_syn_neighbor_distance**| Average distance from the real data to the closest neighbor in the synthetic data|0: all the real rows are leaked in the synthetic dataset. <br/> 1: all the synthetic rows are far away from the real dataset. |
|**close_values_probability**| The probability of close values between the real and synthetic data.| 0: there is no chance to have synthetic rows similar to the real.<br/>1 means that all the synthetic rows are similar to some real rows. |
|**distant_values_probability**| Average distance from the real data to the closest neighbor in the synthetic data|0: no chance to have rows in the synthetic far away from the real data. <br/> 1: all the synthetic datapoints are far away from the real data. |

- __Statistical tests__

| Metric | Description| Values |
|--- | --- | --- |
|**inverse_kl_divergence**|The average inverse of the Kullback–Leibler Divergence| 0: the datasets are from different distributions. <br/> 1: the datasets are from the same distribution.|
|**ks_test**|The Kolmogorov-Smirnov test|0: the distributions are totally different. <br/>1: the distributions are identical.|
|**chi_squared_test**|The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.|0: the distributions are different<br/>1: the distributions are identical.|
|**max_mean_discrepancy**|Empirical maximum mean discrepancy.|0: The distributions are the same. <br/>1: The distributions are totally different.|
|**jensenshannon_dist**|The Jensen-Shannon distance (metric) between two probability arrays. This is the square root of the Jensen-Shannon divergence. |0: The distributions are the same. <br/>1: The distributions are totally different.|
|**wasserstein_dist**| Wasserstein Distance is a measure of the distance between two probability distributions. |0: The distributions are the same.|
|**prdc**| Computes precision, recall, density, and coverage given two manifolds. | --- |
|**alpha_precision**|Evaluate the alpha-precision, beta-recall, and authenticity scores. | --- |
|**survival_km_distance**|The distance between two Kaplan-Meier plots(survival analysis). | --- |
|**fid**|The Frechet Inception Distance (FID) calculates the distance between two distributions of images. | --- |



- __Synthetic Data quality__

| Metric | Description| Values |
|--- | --- | --- |
|**performance.xgb**|Train an XGBoost classifier/regressor/survival model on real data(gt) and synthetic data(syn), and evaluate the performance on the test set. | 1 for ideal performance, 0 for worst performance |
|**performance.linear**|Train a Linear classifier/regressor/survival model on real data(gt) and the synthetic data and evaluate the performance on test data.| 1 for ideal performance, 0 for worst performance |
|**performance.mlp**|Train a Neural Net classifier/regressor/survival model on the real data and the synthetic data and evaluate the performance on test data.| 1 for ideal performance, 0 for worst performance |
|**performance.feat_rank_distance**| Train a model on the synthetic data and a model on the real data. Compute the feature importance of the models on the same test data, and compute the rank distance between the importance(kendalltau or spearman)| 1: similar ranks in the feature importance. 0: uncorrelated feature importance  |
|**detection_gmm**|Train a GaussianMixture model to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|
|**detection_xgb**|Train an XGBoost model to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|
|**detection_mlp**|Train a Neural net to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|
|**detection_linear**|Train a Linear model to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|


- __Privacy metrics__

_Quasi-identifiers_ : pieces of information that are not of themselves unique identifiers, but are sufficiently well correlated with an entity that they can be combined with other quasi-identifiers to create a unique identifier.

| Metric | Description| Values |
|--- | --- | --- |
|**k_anonymization**|The minimum value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.|Reported on both the real and synthetic data.|
|**l_diversity**|The minimum value l which satisfies the l-diversity rule: every generalized block has to contain at least l different sensitive values.|Reported on both the real and synthetic data.|
|**kmap**|The minimum value k which satisfies the k-map rule: every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.|Reported on both the real and synthetic data.|
|**delta_presence**|The maximum re-identification risk for the real dataset from the synthetic dataset.|0 for no risk.|
|**identifiability_score**|The re-identification score on the real dataset from the synthetic dataset.| --- ]
|**sensitive_data_reidentification_xgb**|Sensitive data prediction from the quasi-identifiers using an XGBoost.|0 for no risk.|
|**sensitive_data_reidentification_mlp**|Sensitive data prediction from the quasi-identifiers using a Neural Net.|0 for no risk.|

## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

# Contributing to Synthcity

We want to make contributing to Synthcity is as easy and transparent as possible. We hope to collaborate with as many people as we can.


## Development installation

First create a new environment. It is recommended that you use conda. This can be done as follows:
```bash
conda create -n your-synthcity-env python=3.9
conda activate your-synthcity-env
```
*Python versions 3.7, 3.8, 3.9, and 3.10 are all compatible, but it is best to use the most up to date version you can, as some models may not support older python versions.*

To get the development installation with all the necessary dependencies for
linting, testing, auto-formatting, and pre-commit etc. run the following:
```bash
git clone https://github.com/vanderschaarlab/synthcity.git
cd synthcity
pip install -e .[testing]
```

Please check that the pre-commit is properly installed for the repository, by running:
```bash
pre-commit run --all
```
This checks that you are set up properly to contribute, such that you will match the code style in the rest of the project. This is covered in more detail below.


## Our Development Process

### Code Style

We believe that having a consistent code style is incredibly important. Therefore Synthcity imposes certain rules on the code that is contributed and the automated tests will not pass, if the style is not adhered to. These tests passing is a requirement for a contribution being merged. However, we make adhering to this code style as simple as possible. First, all the libraries required to produce code that is compatible with Synthcity's Code Style are installed in the step above when you set up the development environment. Secondly, these libraries are all triggered by pre-commit, so once you are set-up, you don't need to do anything. When you run `git commit`, any simple changes to enforce the style will run automatically and other required changes are explained in the stdout for you to go through and fix.

Synthcity uses the [black](https://github.com/ambv/black) and [flake8](https://github.com/PyCQA/flake8) code formatter to enforce a common code style across the code base. No additional configuration should be needed (see the [black documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage) for advanced usage).

Also, Synthcity uses [isort](https://github.com/timothycrosley/isort) to sort imports alphabetically and separate into sections.


#### Type Hints

Synthcity is fully typed using python 3.7+ [type hints](https://www.python.org/dev/peps/pep-0484/). This is enforced for contributions by [mypy](https://github.com/python/mypy), which is a static type-checker.


#### Tests

To run the tests, you can either use `pytest` (again, installed with the testing extra above).
The following testing command is good for checking your code,as it skips the tests that take a long time to run.
```bash
pytest -vvvsx -m "not slow" --durations=50
```

But the full test suite can be run with the following command.
```bash
pytest -vvvs  --durations=50
```

Some plugins may be included in the library as extras, the associated tests for these need to be run separately, e.g. the goggle plugin can be tested with the below command:
```bash
pytest -vvvs -k goggle --durations=50
```
## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you have added code that should be tested, add tests in the same style as those already present in the repo.
3. If you have changed APIs, document the API change in the PR.
4. Ensure the test suite passes.
5. Make sure your code passes the pre-commit, this will be required in order to commit and push, if you have properly installed pre-commit, which is included in the testing extra.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## License

By contributing to Synthcity, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree. You should therefore, make sure that if you have introduced any dependencies that they also are covered by a license that allows the code to be used by the project and is compatible with the license in the root directory of this project.

## Citing

If you use this code, please cite the associated paper:

```
@misc{https://doi.org/10.48550/arxiv.2301.07573,
  doi = {10.48550/ARXIV.2301.07573},
  url = {https://arxiv.org/abs/2301.07573},
  author = {Qian, Zhaozhi and Cebere, Bogdan-Constantin and van der Schaar, Mihaela},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Synthcity: facilitating innovative use cases of synthetic data in different data modalities},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

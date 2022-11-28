<h2 align="center">
  synthcity
</h2>

<h4 align="center">
    A library for synthetic data quality assurance.
</h4>


<div align="center">

</div>


Features:
- :key: Easy to extend pluginable architecture.
- :cyclone: Several evaluation metrics for correctness and privacy.
- :fire: Several reference models

## :rocket: Installation

The library can be installed using
```bash
$ pip install .
```

## :boom: Sample Usage

### Generic data
* List the available generators
```python
from synthcity.plugins import Plugins

Plugins(categories=["generic"]).list()
```

* Load and train a generator
```python
from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

syn_model = Plugins().get("marginal_distributions")

syn_model.fit(X)
```

* Generate new synthetic data
```python
syn_model.generate(count = 10)
```

* Generate new synthetic data under some constraints
```python
# Constraint: target <= 100
from synthcity.plugins.core.constraints import Constraints

constraints = Constraints(rules=[("target", "<=", 100)])

generated = syn_model.generate(count=10, constraints=constraints)

assert (generated["target"] <= 100).any()
```

* Benchmark the quality of the plugins
```python
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

loader = GenericDataLoader(X, target_column="target", sensitive_columns=["sex"])

constraints = Constraints(rules=[("target", "ge", 150)])

score = Benchmarks.evaluate(
    ["marginal_distributions"],
    loader,
    synthetic_size=1000,
    synthetic_constraints=constraints,
    metrics={"performance": ["linear_model"]},
    repeats=3,
)
Benchmarks.print(score)
```

### Survival analysis
* List the available generators
```python
from synthcity.plugins import Plugins

Plugins(categories=["survival_analysis"]).list()
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

syn_model = Plugins().get("marginal_distributions")

syn_model.fit(data)

syn_model.generate(count=10)
```

### Time series
* List the available generators
```python
from synthcity.plugins import Plugins

Plugins(categories=["time_series"]).list()
```

* Generate new data
```python
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from synthcity.plugins import Plugins

static_data, temporal_data, outcome = GoogleStocksDataloader().load()
data = TimeSeriesDataLoader(
    temporal_data=temporal_data,
    static_data=static_data,
    outcome=outcome,
)

syn_model = Plugins().get("marginal_distributions")

syn_model.fit(data)

syn_model.generate(count=10)
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

* Using the Serializable interface
```python
from synthcity.plugins import Plugins

syn_model = Plugins().get("adsgan")

buff = syn_model.save()
reloaded = Plugins().load(buff)

assert syn_model.name() == reloaded.name()
```

## 📓 Tutorials
 - [Tutorial 0: Basics](tutorials/tutorial0_basic_examples.ipynb)
 - [Tutorial 1: Write a new plugin](tutorials/tutorial1_add_a_new_plugin.ipynb)
 - [Tutorial 1: Benchmarks](tutorials/tutorial2_benchmarks.ipynb)


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
|**copulagan**| The model is a variation of the CTGAN Model which takes advantage of the CDF based transformation that the GaussianCopulas apply to make the underlying CTGAN model task of learning the data easier.|  |

### Variational autoencoders(VAE)
| Method | Description | Reference |
|--- | --- | --- |
|**tvae**| A conditional VAE network which can handle tabular data.|  [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) |
|**rtvae**| A robust variational autoencoder with β divergence for tabular data (RTVAE) with mixed categorical and continuous features.|  [Robust Variational Autoencoder for Tabular Data with β Divergence](https://arxiv.org/abs/2006.08204) |


### Sampling methods
| Method | Description | Reference |
|--- | --- | --- |
|**marginal_distributions**| A differentially private method that samples from the marginal distributions of the training set|  --- |
|**uniform_sampler**| A differentially private method that uniformly samples from the [min, max] ranges of each column.|  --- |
|**dummy_sampler**| Resample data points from the training set|  --- |

### Normalizing Flows

| Method | Description | Reference |
|--- | --- | --- |
|**nflow**| Normalizing Flows are generative models which produce tractable distributions where both sampling and density evaluation can be efficient and exact.| [Neural Spline Flows](https://arxiv.org/abs/1906.04032) |

### Survival analysis methods
| Method | Description | Reference |
|--- | --- | --- |
|**survival_gan** | SurvivalGAN is a generative model that can handle survival data by addressing the imbalance in the censoring and time horizons, using a dedicated mechanism for approximating time to event/censoring from the input and survival function. | --- |
|**survival_ctgan** | SurvivalGAN version using CTGAN | --- |
|**survae** | SurvivalGAN version using VAE | --- |
|**survival_nflow** | SurvivalGAN version using normalizing flows | --- |

### Time Series methods
| Method | Description | Reference |
|--- | --- | --- |
| **timegan** | TimeGAN is a framework for generating realistic time-series data that combines the flexibility of the unsupervised paradigm with the control afforded by supervised training. Through a learned embedding space jointly optimized with both supervised and adversarial objectives, the network adheres to the dynamics of the training data during sampling.  | [Time-series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) |
| **fflows** |  FFlows is an explicit likelihood model based on a novel class of normalizing flows that view time-series data in the frequency-domain rather than the time-domain. The method uses a discrete Fourier transform (DFT) to convert variable-length time-series with arbitrary sampling periods into fixed-length spectral representations, then applies a (data-dependent) spectral filter to the frequency-transformed time-series.  | [Generative Time-series Modeling with Fourier Flows](https://openreview.net/forum?id=PpshD0AXfA) |
| **probabilistic_ar** | Probabilistic AutoRegressive model allows learning multi-type, multivariate timeseries data and later on generate new synthetic data that has the same format and properties as the learned one. | [PAR model](https://sdv.dev/SDV/user_guides/timeseries/par.html#what-is-par) |


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


## :zap: Evaluation metrics
The following table contains the available evaluation metrics:

#### Sanity checks
| Metric | Description| Values |
|--- | --- | --- |
|**data_mismatch_score**| Average number of columns with datatype(object, real, int) mismatch between the real and synthetic data|0: no datatype mismatch. <br/>1: complete data type mismatch between the datasets.|
|**common_rows_proportion**| The proportion of rows in the real dataset leaked in the synthetic dataset.| 0: there are no common rows between the real and synthetic datasets. <br/> 1: all the rows in the real dataset are leaked in the synthetic dataset. |
|**avg_distance_nearest_neighbor**| Average distance from the real data to the closest neighbor in the synthetic data|0: all the real rows are leaked in the synthetic dataset. <br/> 1: all the synthetic rows are far away from the real dataset. |
|**inlier_probability**| The probability of close values between the real and synthetic data.| 0: there is no chance to have synthetic rows similar to the real.<br/>1 means that all the synthetic rows are similar to some real rows. |
|**outlier_probability**| Average distance from the real data to the closest neighbor in the synthetic data|0: no chance to have rows in the synthetic far away from the real data. <br/> 1: all the synthetic datapoints are far away from the real data. |

#### Statistical tests
| Metric | Description| Values |
|--- | --- | --- |
|**inverse_kl_divergence**|The average inverse of the Kullback–Leibler Divergence| 0: the datasets are from different distributions. <br/> 1: the datasets are from the same distribution.|
|**kolmogorov_smirnov_test**|The Kolmogorov-Smirnov test|0: the distributions are totally different. <br/>1: the distributions are identical.|
|**chi_squared_test**|The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.|0: the distributions are different<br/>1: the distributions are identical.|
|**maximum_mean_discrepancy**|Empirical maximum mean discrepancy.|0: The distributions are the same. <br/>1: The distributions are totally different.|
|**inverse_cdf_distance**|The total distance between continuous features, |0: The distributions are the same. <br/>1: The distributions are totally different.|
|**avg_jensenshannon_distance**|The Jensen-Shannon distance (metric) between two probability arrays. This is the square root of the Jensen-Shannon divergence. |0: The distributions are the same. <br/>1: The distributions are totally different.|
|**feature_correlation**| The correlation/strength-of-association of features in data-set with both categorical and continuous features using: * Pearson's R for continuous-continuous cases * Cramer's V or Theil's U for categorical-categorical cases |0: The distributions are the same. <br/>1: The distributions are totally different.|
|**wasserstein_distance**| Wasserstein Distance is a measure of the distance between two probability distributions. |0: The distributions are the same.|




#### Synthetic Data quality
| Metric | Description| Values |
|--- | --- | --- |
|**train_synth_test_real_data_xbg**|Train an XGBoost classifier or regressor on the synthetic data and evaluate the performance on real test data.|close to 0: similar performance <br/>1: massive performance degradation|
|**train_synth_test_real_data_linear**|Train a Linear classifier or regressor on the synthetic data and evaluate the performance on real test data.|close to 0: similar performance <br/>1: massive performance degradation|
|**train_synth_test_real_data_mlp**|Train a Neural Net classifier or regressor on the synthetic data and evaluate the performance on real test data.|close to 0: similar performance <br/>1: massive performance degradation|
|**gmm_detection**|Train a GaussianMixture model to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|
|**xgb_detection**|Train an XGBoost model to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|
|**mlp_detection**|Train an Neural net to differentiate the synthetic data from the real data.|0: The datasets are indistinguishable. <br/>1: The datasets are totally distinguishable.|


#### Privacy metrics
_Quasi-identifiers_ : pieces of information that are not of themselves unique identifiers, but are sufficiently well correlated with an entity that they can be combined with other quasi-identifiers to create a unique identifier.

| Metric | Description| Values |
|--- | --- | --- |
|**k_anonymization**|The minimum value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.|Reported on both the real and synthetic data.|
|**l_diversity**|The minimum value l which satisfies the l-diversity rule: every generalized block has to contain at least l different sensitive values.|Reported on both the real and synthetic data.|
|**kmap**|The minimum value k which satisfies the k-map rule: every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.|Reported on both the real and synthetic data.|
|**delta_presence**|The maximum re-identification risk for the real dataset from the synthetic dataset.|0 for no risk.|
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

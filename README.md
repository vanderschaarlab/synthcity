<h3 align="center">
  synthcity
</h3>

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
List available generators
```python
from synthcity.plugins import Plugins

Plugins().list()
```

Load and train a generator
```python
from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

syn_model = Plugins().get("dummy_sampler")

syn_model.fit(X)
```

Generate new synthetic data
```python
syn_model.generate(count = 10)
```

Generate new synthetic data under some constraints
```python
# Constraint: target <= 100
from synthcity.plugins.core.constraints import Constraints

constraints = Constraints(rules = [("target", "<=", 100)])

generated = syn_model.generate(count = 10, constraints = constraints)

assert (generated["target"] <= 100).any()
```

Benchmark the quality of the plugins
```python
from synthcity.benchmark import Benchmarks
constraints = Constraints(rules = [("target", "ge", 150)])

score = Benchmarks.evaluate(
    ["dummy_sampler", "random_noise"],
    X, y,
    sensitive_columns = ["sex"],
    synthetic_size = 1000,
    synthetic_constraints = constraints,
    repeats = 5,
)
```

## 📓 Tutorials
 - [Tutorial 0: Basics](tutorials/tutorial0_basic_examples.ipynb)
 - [Tutorial 1: Write a new plugin](tutorials/tutorial1_add_a_new_plugin.ipynb)

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

#### Synthetic Data quality
| Metric | Description| Values |
|--- | --- | --- |
|**train_synth_test_real_data**|Train a classifier or regressor on the synthetic data and evaluate the performance on real test data.|close to 0: similar performance <br/>1: massive performance degradation|
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
|**membership_inference_attack**|The probability of a black-box if a subject from the real subject was used for training the generative model.|0 for no risk.|


## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

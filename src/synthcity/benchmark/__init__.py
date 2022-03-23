# stdlib
from typing import Dict, List, Optional

# third party
import pandas as pd
from IPython.display import display
from pydantic import validate_arguments

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.utils.scores import ScoreEvaluator


class Benchmarks:
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        plugins: List,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_columns: List[str] = [],
        metrics: Optional[Dict] = None,
        repeats: int = 10,
        synthetic_size: Optional[int] = None,
        synthetic_constraints: Optional[Constraints] = None,
    ) -> pd.DataFrame:
        out = {}
        for plugin in plugins:
            scores = ScoreEvaluator()
            for repeat in range(repeats):
                generator = Plugins().get(plugin)

                target_key = f"target_{plugin}_{repeat}"
                X[target_key] = y

                generator.fit(X)

                X_syn = generator.generate(
                    count=synthetic_size, constraints=synthetic_constraints
                )

                evaluation = Metrics.evaluate(
                    X.drop(columns=[target_key]),
                    y,
                    X_syn.drop(columns=[target_key]),
                    X_syn[target_key],
                    sensitive_columns=sensitive_columns,
                    metrics=metrics,
                )

                mean_score = evaluation["mean"].to_dict()
                errors = evaluation["errors"].to_dict()
                duration = evaluation["durations"].to_dict()
                for key in mean_score:
                    scores.add(key, mean_score[key], errors[key], duration[key])
            out[plugin] = scores.to_dataframe()

        return out

    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def print(
        results: Dict,
    ) -> None:
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        for plugin in results:
            print()
            print("\033[4m" + "\033[1m" + f"Plugin : {plugin}" + "\033[0m" + "\033[0m")
            display(results[plugin])
            print()

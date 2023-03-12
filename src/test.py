# third party
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# loadDebugger()
X, y = load_iris(as_frame=True, return_X_y=True)
X = GenericDataLoader(X.assign(target=y), target_column="target")
plugin = Plugins().get(
    "ddpm",
    n_iter=3,
    is_classification=True,
    gaussian_loss_type="mse",
    num_timesteps=100,
    verbose=1,
)
plugin.fit(X)
X_syn = plugin.model.generate(50)

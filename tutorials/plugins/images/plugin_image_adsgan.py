# %% [markdown]
# # Image AdsGAN Example

# %%
# stdlib
import warnings

warnings.filterwarnings("ignore")

# synthcity absolute
from synthcity.plugins import Plugins

eval_plugin = "image_adsgan"

# %% [markdown]
# ### Load dataset
#

# %%
# third party
from torchvision import datasets, transforms

# synthcity absolute
from synthcity.plugins.core.dataloader import ImageDataLoader

IMG_SIZE = 32

dataset = datasets.MNIST(".", download=True)
loader = ImageDataLoader(
    dataset,
    height=IMG_SIZE,
).sample(1000)

loader.shape

# %% [markdown]
# ### Train the generator
#

# %%
# synthcity absolute
from synthcity.plugins import Plugins

syn_model = Plugins().get(eval_plugin, batch_size=100, plot_progress=True, n_iter=100)

syn_model.fit(loader)

# %% [markdown]
# ### Generate new samples
#

# %%
# third party
import torch

# synthcity absolute
from synthcity.plugins.core.models.image_gan import display_imgs

syn_samples, syn_labels = syn_model.generate(count=50).unpack().tensors()

display_imgs(syn_samples)

# %% [markdown]
# ### Benchmarks

# %%
# synthcity absolute
from synthcity.benchmark import Benchmarks

score = Benchmarks.evaluate(
    [
        (eval_plugin, eval_plugin, {"n_iter": 50})
    ],  # (testname, plugin, plugin_args) REPLACE {"n_iter" : 50} with {} for better performance
    loader,
    repeats=2,
    metrics={"detection": ["detection_mlp"]},  # DELETE THIS LINE FOR ALL METRICS
    task_type="classification",
)

# %%
Benchmarks.print(score)

# %%

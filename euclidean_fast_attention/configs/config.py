"""The default configuration for the experiments."""

import ml_collections
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.seed = 42

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = 'euclidean_fast_attention'
    config.wandb.group = config_dict.placeholder(str)
    config.wandb.name = config_dict.placeholder(str)

    return config

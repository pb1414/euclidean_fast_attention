"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = config_dict.placeholder(int)
    config.num_valid = 500
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = config_dict.placeholder(int)
    config.max_num_edges = config_dict.placeholder(int)
    config.max_num_graphs = config_dict.placeholder(int)
    config.num_epochs = 1000
    config.num_train_steps = config_dict.placeholder(int)
    config.save_interval_steps = 5000
    config.log_loss_every_steps = 500
    config.energy_unit = units.kcal / units.mol
    config.length_unit = units.Angstrom
    config.log_loss_every_steps = 50
    config.pbc_bool = False
    config.auto_eval = True

    return config

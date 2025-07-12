"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 8_000
    config.num_valid = 500
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 8 * 56 + 1
    config.max_num_edges = 8 * 56 * 30 + 1  # works for cutoff 4.0, 5.0, 6.0
    config.max_num_graphs = 8 + 1
    config.num_epochs = None
    config.num_train_steps = 1_000_000
    config.save_interval_steps = 5000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.log_loss_every_steps = 50
    config.pbc_bool = False
    config.auto_eval = True
    config.subtract_energy_mean = True
    config.energy_weight = 0.01
    config.forces_weight = 0.99
    config.neighbor_list_cutoff = config_dict.placeholder(float)

    return config

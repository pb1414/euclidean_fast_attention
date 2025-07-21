"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict
from euclidean_fast_attention.configs.lookup import get_avg_num_neighbors_md22


def get_config(cutoff: str = '4'):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 6_000
    config.num_valid = 500
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 8 * 42 + 1
    config.max_num_edges = 8 * 42 * get_avg_num_neighbors_md22(split='AcAla3NHMe', cutoff=int(cutoff)) + 1
    config.max_num_graphs = 8 + 1
    config.num_epochs = None
    config.num_train_steps = 1_000_000
    config.save_interval_steps = 5000
    config.log_loss_every_steps = 50
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.pbc_bool = False
    config.auto_eval = True
    config.subtract_energy_mean = True
    config.energy_weight = 0.001
    config.forces_weight = 0.999
    config.neighbor_list_cutoff = config_dict.placeholder(float)

    return config

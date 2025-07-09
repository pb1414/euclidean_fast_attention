"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config(num_atoms: str):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 4000
    config.num_valid = 500
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 16 * int(num_atoms) + 1
    config.max_num_edges = 16 * int(num_atoms) * avg_num_neighbors_lookup[int(num_atoms)] + 1
    config.max_num_graphs = 16 + 1
    config.num_epochs = None
    config.num_train_steps = 500_000
    config.save_interval_steps = 5_000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.pbc_bool = False
    config.auto_eval = True
    config.subtract_energy_mean = True
    config.neighbor_list_cutoff = config_dict.placeholder(float)

    return config


# assumes a model cutoff of 5.0 Angstrom
avg_num_neighbors_lookup = {
    16: 3,
    32: 3,
    64: 3,
    128: 3,
    256: 5,
}

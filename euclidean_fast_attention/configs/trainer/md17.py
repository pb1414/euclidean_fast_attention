"""The default configuration for the experiments."""

import ml_collections

from ase import units
from ml_collections import config_dict


def get_config(split: str):
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.datafile = config_dict.placeholder(str)
    config.num_train = 1000
    config.num_valid = 500
    config.split_seed = 0
    config.model_seed = 0
    config.max_num_nodes = 4 * number_of_atoms_lookup[split] + 1
    config.max_num_edges = 4 * number_of_atoms_lookup[split] * (number_of_atoms_lookup[split] - 1) + 1
    config.max_num_graphs = 4 + 1
    config.num_epochs = None
    config.num_train_steps = 1_250_000
    config.save_interval_steps = 5000
    config.log_loss_every_steps = 500
    config.energy_unit = units.eV
    config.length_unit = units.Angstrom
    config.log_loss_every_steps = 50
    config.pbc_bool = False
    config.auto_eval = True
    config.subtract_energy_mean = True

    return config

number_of_atoms_lookup = {
    'aspirin': 21,
    'toluene': 15,
    'uracil': 12,
    'naphthalene': 18,
    'salicylic': 16,
    'malonaldehyde': 9,
    'ethanol': 9,
}
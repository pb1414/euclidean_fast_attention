"""The default configuration for the experiments."""

import numpy as np
import ml_collections

from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.name = 'base_model'
    config.cutoff = 4.0  # in Angstrom
    config.num_features = 64
    config.num_layers = 3
    config.mp_max_degree = 1
    config.mp_num_basis_fn = 32
    config.radial_basis_fn = 'exponential_bernstein'
    config.emulate_era_block = False
    config.era_use_in_iterations = "1"
    config.era_max_degree = 1
    config.era_include_pseudotensors = False
    config.era_tensor_integration = True
    config.era_ti_max_degree_sph = 1
    config.era_ti_max_degree = 1
    config.era_ti_parametrize_coupling_paths = True
    config.era_activation_fn = "identity"
    config.era_num_frequencies = None
    config.era_max_frequency = np.pi
    config.era_max_length = 11.0
    config.era_lebedev_num = 50
    config.era_qk_num_features = 16
    config.era_v_num_features = 16
    config.num_post_residual_mlps = 0
    config.use_switch = False
    config.iterated_tensor_products = False

    return config

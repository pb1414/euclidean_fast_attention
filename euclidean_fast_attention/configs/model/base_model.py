"""The default configuration for the experiments."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.name = 'base_model'

    config = ml_collections.ConfigDict()
    config.cutoff = 5.0  # in Angstrom
    config.num_features = 64
    config.num_iterations = 3
    config.mp_max_degree = 2
    config.mp_num_basis_fn = 32
    config.era_use_in_iterations = "0 1"
    config.era_max_degree = 1
    config.era_include_pseudotensors = False
    config.era_activation_fn = "gelu"
    config.era_num_frequencies = None
    config.era_max_frequency = 3.141592653589793
    config.era_max_length = -1.
    config.era_lebedev_num = 50
    config.era_qk_num_features = 16
    config.era_v_num_features = 32
    config.num_post_residual_mlps = 0
    config.use_switch = False
    config.emulate_era_block = False
    config.iterated_tensor_products = False

    return config

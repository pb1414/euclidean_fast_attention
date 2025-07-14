"""The default configuration for the experiments."""

import numpy as np
import ml_collections


def get_config(num_atoms: str):
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()

    config.name = 'base_model'
    config.cutoff = 5.0  # in Angstrom
    config.num_features = 128
    config.num_layers = 3
    config.mp_max_degree = 0
    config.mp_num_basis_fn = 32
    config.radial_basis_fn = 'reciprocal_bernstein'
    config.emulate_era_block = False
    config.era_use_in_iterations = "0 1"
    config.era_max_degree = 0
    config.era_include_pseudotensors = False
    config.era_activation_fn = 'identity'
    config.era_num_frequencies = None
    config.era_max_frequency = float(3*np.pi)
    config.era_max_length = 50.0
    config.era_lebedev_num = 146
    config.era_qk_num_features = calculate_num_features(num_atoms)
    config.era_v_num_features = calculate_num_features(num_atoms)
    config.num_post_residual_mlps = 0
    config.use_switch = False
    config.iterated_tensor_products = False
    config.dispersion_correction_bool = False
    
    return config

def calculate_num_features(num_atoms: str, base_num_features: int = 16, base_num_atoms: int = 16):
    num_atoms_int = int(num_atoms)

    # Logarithmically increase the features with the number of atoms and round to the next integer
    num_features = np.floor(
        1 + base_num_features * np.log(num_atoms_int / base_num_atoms)
    ).item()

    num_features = int(num_features)

    # Ensure num_features is always even
    if num_features % 2 != 0:
        num_features += 1

    return num_features

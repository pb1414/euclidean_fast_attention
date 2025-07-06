"""The default configuration for the experiments."""

import numpy as np

from ml_collections import config_dict
from euclidean_fast_attention.configs.lookup import molecular_graph_lookup, max_length_lookup


def get_config():
    """Get the default hyperparameter configuration."""

    config = config_dict.ConfigDict()

    config.name = 'schnet'

    # Model Architecture Parameters
    config.num_layers = 3
    config.num_features = 128

    # Interaction Parameters
    config.cutoff = 4.0
    
    # Radial Basis Function Parameters
    config.radial_basis_fn = 'exponential_bernstein'
    config.num_basis_fn = 32

    # Atomic Number / Element Range
    config.zmax = 119

    # Euclidean Fast Attention (EFA) Block Parameters
    config.use_efa_block = True
    config.emulate_efa_block = False
    config.era_max_length = 10.0
    config.era_max_frequency = float(np.pi)
    config.era_qk_num_features = 16
    config.era_v_num_features = 32
    config.era_lebedev_num = 50
    config.efa_block_behaves_like_identity_at_init = True
    
    return config

"""Functions that create the building blocks from configDict."""

import e3x
import ml_collections
import optax

from euclidean_fast_attention import NpzTrainer
from euclidean_fast_attention import model
from euclidean_fast_attention import schnet

def create_optimizer_from_config(
        config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:
    """Creates an optimizer, as specified by the config."""
    if config.optimizer.schedule == "constant":
        schedule = optax.constant_schedule(config.optimizer.learning_rate)
    elif config.optimizer.schedule == "cosine_decay":
        schedule = optax.cosine_decay_schedule(
            config.optimizer.learning_rate,
            decay_steps=config.trainer.num_train_steps,
        )
    elif config.optimizer.schedule == "exponential_decay":
        decay_rate = (
                config.optimizer.stop_learning_rate / config.optimizer.learning_rate
        )
        schedule = optax.exponential_decay(
            config.optimizer.learning_rate,
            transition_steps=config.trainer.num_train_steps,
            decay_rate=decay_rate,
        )
    else:
        raise ValueError(f"Unknown schedule: {config.optimizer.schedule}.")
    base_optimizer_fn = getattr(optax, config.optimizer.name)

    maybe_clip_by_global_norm = (
        optax.clip_by_global_norm(config.optimizer.clip_by_global_norm)
        if config.optimizer.clip_by_global_norm is not None
        else optax.identity()
    )

    @optax.inject_hyperparams
    def optimizer_fn(learning_rate):
        chain = [
            optax.zero_nans(),  # TODO(b/296999153) Can we avoid this?
            maybe_clip_by_global_norm,
            base_optimizer_fn(learning_rate=learning_rate),
        ]
        return optax.chain(*chain)

    tx = optimizer_fn(learning_rate=schedule)
    return tx


def create_base_model_from_config(config: ml_collections.ConfigDict):
    """Creates an energy model, as specified by the config.

  Args:
    config: `ConfigDict`.

  Returns:

  """
    model_config = dict(config.model)

    if model_config["era_activation_fn"] == "identity":
        era_activation_fn = lambda u: u
    else:
        era_activation_fn = getattr(e3x.nn, model_config["era_activation_fn"])

    era_use_in_iterations = model_config["era_use_in_iterations"].split()
    era_use_in_iterations = list(map(int, era_use_in_iterations))
    if len(era_use_in_iterations) == 0:
        era_use_in_iterations = None
    else:
        # era_use_in_iterations starts at index 0.
        assert max(era_use_in_iterations) < model_config["num_layers"]

    model_config["era_activation_fn"] = era_activation_fn
    model_config["era_use_in_iterations"] = era_use_in_iterations
    model_config['pbc_bool'] = config.trainer.pbc_bool

    return model.EnergyModel(**model_config)


def create_schnet_from_config(config: ml_collections.ConfigDict):
    model_config = dict(config.model)
    model_config['pbc_bool'] = config.trainer.pbc_bool
    
    return schnet.SchNet(**model_config)


def create_trainer_from_config(config: ml_collections.ConfigDict):
    """Creates a trainer, as specified by the config.

      Args:
        config: `ConfigDict`.

      Returns:

      Raises:
        ValueError:
      """

    return NpzTrainer(
        data_dir=config.trainer.datafile,
        num_train=config.trainer.num_train,
        num_valid=config.trainer.num_valid,
        split_seed=config.trainer.split_seed,
        model_seed=config.trainer.model_seed,
        num_epochs=config.trainer.num_epochs,
        save_interval_steps=config.trainer.save_interval_steps,
        max_num_nodes=config.trainer.max_num_nodes,
        max_num_edges=config.trainer.max_num_edges,
        max_num_graphs=config.trainer.max_num_graphs,
        energy_unit=config.trainer.energy_unit,
        length_unit=config.trainer.length_unit,
        pbc_bool=config.trainer.pbc_bool,
        subtract_energy_mean=config.trainer.subtract_energy_mean,
        neighbor_list_cutoff=config.trainer.neighbor_list_cutoff,
    )


def default_num_train(config):
    molecule_name = config.trainer.split
    return num_train_lookup[molecule_name]


def calculate_batch_configs(
        config, capacity_multiplier: float = 1.1
):
    max_num_graphs = config.trainer["max_num_graphs"]
    molecule_name = config.trainer.split
    cutoff = config.model.cutoff
    num_nodes, avg_num_neighbors = lookup[cutoff][molecule_name]
    max_num_nodes = (max_num_graphs - 1) * num_nodes + 1
    max_num_edges = (max_num_graphs - 1) * avg_num_neighbors * num_nodes
    max_num_edges = int(max_num_edges * capacity_multiplier)
    return max_num_nodes, max_num_edges


def get_max_length(config):
    molecule_name = config.trainer.split
    return max_length_lookup[molecule_name]


max_length_lookup = {
    "ethanol": 10.0,  # (number of atoms, max length in data)
    "aspirin": 10.0,
    "toluene": 10.0,
    "uracil": 10.0,
    "naphthalene": 10.0,
    "salicylic": 10.0,
    "malonaldehyde": 10.0,
    "AT-AT": 22.0,
    "AT-AT-CG-CG": 24.0,
    "Ac-Ala3-NHMe": 12.0,
    "DHA": 16.0,
    "buckyball-catcher": 15.0,
    "double-walled_nanotube": 33.0,
    "stachyose": 14.0,
}

lookup_cutoff4 = {  # (number of atoms, avg. number of neighbors per atom = max # total neighbors / num_atoms)
    "AT-AT": (60, 15.666666666666666),
    "AT-AT-CG-CG": (118, 17.220338983050848),
    "Ac-Ala3-NHMe": (42, 16.857142857142858),
    "DHA": (56, 18.0),
    "buckyball-catcher": (148, 17.81081081081081),
    "double-walled_nanotube": (370, 24.67027027027027),
    "stachyose": (87, 21.563218390804597),
}

lookup_cutoff5 = {
    "ethanol": (9, 8),  # (number of atoms, avg. # of neighbors per atom = max # total neighbors / num_atoms)
    "aspirin": (21, 20),
    "toluene": (15, 14),
    "uracil": (12, 11),
    "naphthalene": (18, 17),
    "salicylic": (16, 15),
    "malonaldehyde": (9, 8),
    "AT-AT": (60, 25.466666666666665),
    "AT-AT-CG-CG": (118, 30.1864406779661),
    "Ac-Ala3-NHMe": (42, 26.61904761904762),
    "DHA": (56, 28.5),
    "buckyball-catcher": (148, 32.310810810810814),
    "double-walled_nanotube": (370, 44.28108108108108),
    "stachyose": (87, 35.12643678160919),
}

lookup = {4: lookup_cutoff4, 5: lookup_cutoff5}

num_train_lookup = {
    "ethanol": 1000,
    "aspirin": 1000,
    "toluene": 1000,
    "uracil": 1000,
    "naphthalene": 1000,
    "salicylic": 1000,
    "malonaldehyde": 1000,
    "AT-AT": 3000,
    "AT-AT-CG-CG": 2000,
    "Ac-Ala3-NHMe": 6000,
    "DHA": 8000,
    "buckyball-catcher": 600,
    "double-walled_nanotube": 800,
    "stachyose": 8000,
}


def get_number_of_epochs(config, target_steps: int, min_num_epochs=250):
    num_train = config.trainer.num_train
    batch_size = config.trainer.max_num_graphs - 1
    return max(min_num_epochs, int(target_steps / (num_train / batch_size)))


def get_number_of_train_steps(config, target_epochs: int):
    num_train = config.trainer.num_train
    batch_size = config.trainer.max_num_graphs - 1
    return int(target_epochs * (num_train / batch_size))

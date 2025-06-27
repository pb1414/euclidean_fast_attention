"""Script for running the training of a model."""

import json

from clu import metric_writers
import pathlib
import jax
import ml_collections
import wandb
from orbax import checkpoint

from euclidean_fast_attention.utils import from_config


def run_training(config: ml_collections.ConfigDict, workdir):
    """Train and evaluate a model, given `config` and write to `workdir`.

  Args:
    config:
    workdir:
  """
    workdir = pathlib.Path(workdir)
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)

    if config.trainer.num_train < 0:
        raise ValueError(
            f'num_train must be larger than 0. Value is {config.trainer.num_train}.'
        )

    if config.trainer.max_num_nodes < 0:
        raise ValueError(
            f'max_num_nodes must be larger than 0. Value is {config.trainer.max_num_nodes}.'
        )

    if config.trainer.max_num_edges < 0:
        raise ValueError(
            f'max_num_edges must be larger than 0. Value is {config.trainer.max_num_edges}.'
        )
    
    if config.trainer.num_epochs is None:
        assert config.trainer.num_train_steps is not None, 'One of num_epochs or num_train_steps must be set.'

        num_epochs = from_config.get_number_of_epochs(
            config, target_steps=config.trainer.num_train_steps
        )
        config.trainer['num_epochs'] = num_epochs
    
    if config.trainer.num_train_steps is None:
        assert config.trainer.num_epochs is not None, 'One of num_epochs or num_train_steps must be set.'

        num_train_steps = from_config.get_number_of_train_steps(
            config, target_epochs=config.trainer.num_epochs
        )
        config.trainer['num_train_steps'] = num_train_steps

    optimizer = from_config.create_optimizer_from_config(config=config)
    
    if config.model.name == 'base_model':
        if config.model.era_use_in_iterations is not None:
            if config.model.era_max_length is None:
                raise ValueError(
                    f'era_max_length must be specified. Received "None".'
                )
        energy_model = from_config.create_base_model_from_config(config)
    elif config.model.name == 'schnet':
        if config.model.use_efa_block is True:
            if config.model.era_max_length is None:
                raise ValueError(
                    f'era_max_length must be specified. Received "None".'
                )
        energy_model = from_config.create_schnet_from_config(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model.name}")

    trainer = from_config.create_trainer_from_config(config=config)

    config_as_json = config.to_dict()
    with open(workdir / 'config.json', 'w') as f:
        json.dump(config_as_json, f)

    _ = trainer.run_training(
        model=energy_model, optimizer=optimizer, ckpt_dir=workdir / 'checkpoint'
    )

    # After training has finished, load the parameters from the best checkpoint.
    loaded_mngr = checkpoint.CheckpointManager(
        workdir / 'checkpoint',
        {
            'params': checkpoint.PyTreeCheckpointer(),
            'opt_state': checkpoint.PyTreeCheckpointer(),
        },
        options=checkpoint.CheckpointManagerOptions(step_prefix='ckpt'),
    )
    mgr_state = loaded_mngr.restore(loaded_mngr.latest_step())
    params = mgr_state.get('params')

    # Calculate the metrics on the test split.
    test_metrics, _ = trainer.run_testing(
        model=energy_model,
        params=params,
        collect_predictions=False
    )

    # Dump the test metrics to json.
    with open(workdir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

"""Script for running the training of a model."""

import json

import pathlib
import ml_collections
import logging
import numpy as np
from orbax import checkpoint
from typing import Optional

from euclidean_fast_attention.utils import from_config


def run_evaluation(
    workdir: str, 
    eval_name: str,
    datafile: Optional[str] = None,
    collect_predictions: bool = False
):
    """Evaluate a model, given `workdir` and `eval_name`.

    Args:
        workdir: Path to the workdir.
        eval_name: Name of the evaluation.
    """
    workdir = pathlib.Path(workdir)
    workdir = workdir.resolve()

    with open(workdir / 'config.json', 'r') as f:
        config = ml_collections.ConfigDict(json.load(f))


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
    
    if datafile is not None:
        new_datafile = str(pathlib.Path(datafile).resolve())
        print(
            f'Overwrite datafile in config {config.trainer.datafile} with {new_datafile}.'
            f'All data in this datafile will be used for evaluation.'
        )
        config.trainer.datafile = new_datafile
        
        if config.trainer.subtract_energy_mean == True:
            logging.warning(
                'subtract_energy_mean was set to True during training. For evaluation on a different datafile, the energy mean might not be transferable.'
                'Thus, energy values might be off by a constant yielding high errors.'
            )
        
        config.trainer.subtract_energy_mean = False
        config.trainer.num_train = 0
        config.trainer.num_valid = 0
        
        # TODO: Here we might need also adapt the max_num_graphs, max_num_nodes, max_num_edges.
        trainer = from_config.create_trainer_from_config(config=config)
    else:
        trainer = from_config.create_trainer_from_config(config=config)


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
    test_metrics, (energy_predictions, forces_predictions, energy_gt, forces_gt, graphs) = trainer.run_testing(
        model=energy_model,
        params=params,
        collect_predictions=collect_predictions,
        
    )

    # Dump the test metrics to json.
    with open(workdir / f'{eval_name}_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

    if collect_predictions == True:
        energy_predictions = np.concatenate(energy_predictions, axis=0)
        forces_predictions = np.concatenate(forces_predictions, axis=0)
        energy_gt = np.concatenate(energy_gt, axis=0)
        forces_gt = np.concatenate(forces_gt, axis=0)

        np.savez(
            f'{workdir}/predictions_{eval_name}.npz',
            energy_predictions=energy_predictions,
            forces_predictions=forces_predictions,
            energy_gt=energy_gt,
            forces_gt=forces_gt,
        )

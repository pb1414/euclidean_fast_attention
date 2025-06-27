"""Trainer for regular npz files."""

import dataclasses
import functools
import jax
import jax.numpy as jnp
import jaxtyping
import jraph
import numpy as np
import random
import wandb

from orbax import checkpoint
from typing import Any

from . import training_utils
from . import jraph_utils


Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer


# same method as for tfds data set
def remove_energy_offset(
        element: dict[str, Any], energy_shift: Float
) -> dict[str, Any]:
    element['energy'] = element['energy'] - energy_shift
    return element


def convert_to_electron_volt(
        element: dict[str, Any], conversion_factor
) -> dict[str, Any]:
    element['energy'] = element['energy'] * conversion_factor
    element['forces'] = element['forces'] * conversion_factor
    return element


def convert_to_angstrom(
        element: dict[str, Any], conversion_factor
) -> dict[str, Any]:
    element['forces'] = element['forces'] * 1 / conversion_factor
    element['coordinates'] = element['coordinates'] * conversion_factor
    
    if element['lattice_vectors'] is not None:
        element['lattice_vectors'] = element['lattice_vectors'] * conversion_factor
    
    return element


def split_into_train_valid_test(
        data_dir,
        num_train: int,
        num_valid: int,
        split_seed: int = 0,
        shuffle: bool = True
):
    """Split into training, validation and test.

      Args:
        data_dir: Path to the `.npz` file.
        num_train: Number of training points.
        num_valid: Number of test points.
        split_seed: Seed to use for split.
        shuffle: Shuffle the data.

      Returns:

      """
    np.random.seed(split_seed)

    with np.load(data_dir) as f:
        positions = f['positions']

        if num_train + num_valid > len(positions):
            raise ValueError(
                f'{num_train=} and {num_valid=} exceeds the cardinality'
                f' of the dataset (dataset_size={len(positions)}).'
            )
        if shuffle:
            random_permutation = np.random.permutation(np.arange(len(positions)))
        else:
            random_permutation = np.arange(len(positions))

        atomic_numbers = f['atomic_numbers']
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.reshape(-1, 1)
            atomic_numbers = np.repeat(atomic_numbers, len(positions), axis=0)

        energy = f['energy']
        if energy.ndim == 2:
            energy = energy.reshape(-1)

        forces = f['forces']

        try:
            atomic_dipoles = f['atomic_dipoles']
        except KeyError:
            atomic_dipoles = None

        try:
            node_mask = f['node_mask']
        except KeyError:
            node_mask = None

        try:
            lattice_vectors = f['lattice_vectors']
        except KeyError:
            lattice_vectors = None

        positions = positions[random_permutation]
        atomic_numbers = atomic_numbers[random_permutation]
        energy = energy[random_permutation]
        forces = forces[random_permutation]

        if atomic_dipoles is not None:
            atomic_dipoles = atomic_dipoles[random_permutation]

        if node_mask is not None:
            node_mask = node_mask[random_permutation]
        
        if lattice_vectors is not None:
            lattice_vectors = lattice_vectors[random_permutation]

        energy_mean = np.mean(energy[:num_train])

        # we call positions coordinates here, to be consistent with the format in
        # the qcml tfds data sets
        train_data = iter(
            [
                {
                    'coordinates': np.asarray(positions[i]),
                    'lattice_vectors': np.asarray(lattice_vectors[i]) if lattice_vectors is not None else None,
                    'atomic_numbers': np.asarray(
                        atomic_numbers[i], dtype=np.int32
                    ),
                    'atomic_dipoles': np.asarray(atomic_dipoles[i]) if atomic_dipoles is not None else None,
                    'energy': np.asarray(energy[i]),
                    'forces': np.asarray(forces[i]),
                    'node_mask': np.asarray(node_mask[i]) if node_mask is not None else None
                }
                for i in np.arange(num_train)
            ]
        )
        valid_data = iter(
            [
                {
                    'coordinates': np.asarray(positions[i]),
                    'lattice_vectors': np.asarray(lattice_vectors[i]) if lattice_vectors is not None else None,
                    'atomic_numbers': np.asarray(
                        atomic_numbers[i], dtype=np.int32
                    ),
                    'atomic_dipoles': np.asarray(atomic_dipoles[i]) if atomic_dipoles is not None else None,
                    'energy': np.asarray(energy[i]),
                    'forces': np.asarray(forces[i]),
                    'node_mask': np.asarray(node_mask[i]) if node_mask is not None else None
                }
                for i in np.arange(num_train, num_train + num_valid)
            ]
        )
        test_data = iter(
            [
                {
                    'coordinates': np.asarray(positions[i]),
                    'lattice_vectors': np.asarray(lattice_vectors[i]) if lattice_vectors is not None else None,
                    'atomic_numbers': np.asarray(
                        atomic_numbers[i], dtype=np.int32
                    ),
                    'atomic_dipoles': np.asarray(atomic_dipoles[i]) if atomic_dipoles is not None else None,
                    'energy': np.asarray(energy[i]),
                    'forces': np.asarray(forces[i]),
                    'node_mask': np.asarray(node_mask[i]) if node_mask is not None else None
                }
                for i in np.arange(num_train + num_valid, len(positions))
            ]
        )

        return train_data, valid_data, test_data, energy_mean


@dataclasses.dataclass
class NpzTrainer:
    """Trainer for regular npz files.

    Attributes:
    data_dir:
    num_train:
    num_valid:
    num_epochs:
    max_num_nodes:
    max_num_edges:
    max_num_graphs:
    energy_unit:
    length_unit:
    save_interval_steps:
    split_seed:
    model_seed:
    """
    data_dir: str

    num_train: int
    num_valid: int

    num_epochs: int

    max_num_nodes: int
    max_num_edges: int
    max_num_graphs: int

    energy_unit: float
    length_unit: float

    save_interval_steps: int
    log_interval_steps: int = 1

    split_seed: int = 0
    model_seed: int = 0

    energy_weight: float = 0.01
    forces_weight: float = 0.99

    subtract_energy_mean: bool = True

    use_wandb: bool = True

    pbc_bool: bool = False

    def prepare_training_and_validation_data(self, cutoff=None, shuffle=True):
        """Prepare training and validation data.

    Args:
      cutoff: Cutoff used for edge computation.
      shuffle: Shuffle the data.

    Returns:

    """

        train_split, valid_split, _, energy_mean = split_into_train_valid_test(
            self.data_dir, num_train=self.num_train, num_valid=self.num_valid, shuffle=shuffle,
        )
        if self.subtract_energy_mean:
            print(f'Subtract energy mean={energy_mean:.4f}.')
        else:
            energy_mean = 0.
            print(f'Do not subtract energy mean.')

        train_split = map(
            functools.partial(
                convert_to_angstrom, conversion_factor=self.length_unit
            ),
            map(
                functools.partial(
                    convert_to_electron_volt, conversion_factor=self.energy_unit
                ),
                map(
                    functools.partial(
                        remove_energy_offset, energy_shift=energy_mean
                    ),
                    train_split,
                ),
            ),
        )

        valid_split = map(
            functools.partial(
                convert_to_angstrom, conversion_factor=self.length_unit
            ),
            map(
                functools.partial(
                    convert_to_electron_volt, conversion_factor=self.energy_unit
                ),
                map(
                    functools.partial(
                        remove_energy_offset, energy_shift=energy_mean
                    ),
                    valid_split,
                ),
            ),
        )

        # graph construction configs
        if cutoff is None:
            print(
                'No cutoff specified for graph construction. This can lead to'
                ' computational overhead.'
            )
            cutoff = 1e6

        make_graph_tuple = functools.partial(
            jraph_utils.create_graph_tuple, cutoff=cutoff, pbc_bool=self.pbc_bool
        )

        prepared_train_ds = map(make_graph_tuple, train_split)
        prepared_valid_ds = map(make_graph_tuple, valid_split)

        return prepared_train_ds, prepared_valid_ds

    def prepare_testing_data(self, cutoff=None, shuffle=True, subtract_energy_mean: bool = True):
        """Prepare testing data.

    Args:
      cutoff: Cutoff used for edge computation.
      shuffle: Shuffle data.
      subtract_energy_mean: Subtract energy mean.

    Returns:
    """

        _, _, test_split, energy_mean = split_into_train_valid_test(
            self.data_dir, num_train=self.num_train, num_valid=self.num_valid, shuffle=shuffle
        )

        if not subtract_energy_mean:
            energy_mean = 0.

        test_split = map(
            functools.partial(
                convert_to_angstrom, conversion_factor=self.length_unit
            ),
            map(
                functools.partial(
                    convert_to_electron_volt, conversion_factor=self.energy_unit
                ),
                map(
                    functools.partial(
                        remove_energy_offset, energy_shift=energy_mean
                    ),
                    test_split,
                ),
            ),
        )

        # graph construction configs
        if cutoff is None:
            print(
                'No cutoff specified for graph construction. This can lead to'
                ' computational overhead.'
            )
            cutoff = 1e6

        make_graph_tuple = functools.partial(
            jraph_utils.create_graph_tuple, cutoff=cutoff, pbc_bool=self.pbc_bool
        )

        prepared_test_ds = map(make_graph_tuple, test_split)

        return prepared_test_ds

    def init_ckpt_manager(self, ckpt_dir, ckpt_manager_options=None):
        """Initialize `orbax.ChekpointManager`.

    Args:
      ckpt_dir: Checkpoint directory.
      ckpt_manager_options: `orbax.CheckpointManagerOptions`.

    Returns:
    """
        if ckpt_dir is None:
            return None
        else:
            if ckpt_manager_options is None:
                ckpt_manager_options = {
                    'max_to_keep': 1,
                    'save_interval_steps': self.save_interval_steps,
                }

            options = checkpoint.CheckpointManagerOptions(
                best_fn=lambda u: u['loss'],
                best_mode='min',
                step_prefix='ckpt',
                **ckpt_manager_options,
            )

            mngr = checkpoint.CheckpointManager(
                ckpt_dir,
                {
                    'params': checkpoint.PyTreeCheckpointer(),
                    'opt_state': checkpoint.PyTreeCheckpointer(),
                },
                options=options,
            )

            return mngr

    def run_testing(
            self,
            params,
            model,
            num_test=None,
            collect_predictions=False,
            max_num_graphs=None,
            max_num_nodes=None,
            max_num_edges=None,
    ):
        """Evaluate a `model` with `params` on the `NpzTrainer` test split.

        Args:
          params:
          model:
          num_test:
          collect_predictions:
          max_num_graphs:
          max_num_nodes:
          max_num_edges:

        Returns:

        """
        num_graphs = (
            self.max_num_graphs if max_num_graphs is None else max_num_graphs
        )
        num_nodes = self.max_num_nodes if max_num_nodes is None else max_num_nodes
        dynamically_batch = functools.partial(
            jraph.dynamically_batch,
            n_node=num_nodes,
            n_edge=self.max_num_edges if max_num_edges is None else max_num_edges,
            n_graph=num_graphs,
        )
        jitted_batch_segments_fn = jax.jit(jraph_utils.batch_segments_fn)

        inference_fn = jax.jit(training_utils.make_inference_fn(params, model))
        prepared_test_ds = self.prepare_testing_data(
            model.cutoff,
            subtract_energy_mean=self.subtract_energy_mean
        )
        test_iter = dynamically_batch(iter(prepared_test_ds))
        # eval_metrics_list = []

        energy_predictions = []
        forces_predictions = []
        energy_gt = []
        forces_gt = []
        graphs = []
        energy_mae = []
        forces_mae = []
        energy_mse = []
        forces_mse = []
        running_num_of_evaluated_structures = 0
        for n, graph_batch in enumerate(test_iter):
            batch_segments_dict = jitted_batch_segments_fn(graph_batch)
            inputs = jraph_utils.jraph_to_input(graph_batch)
            inputs.update(batch_segments_dict)

            energy, forces = inference_fn(inputs)
            energy_mae += [training_utils.mean_absolute_error(
                energy, inputs['energy'], msk=batch_segments_dict['graph_mask']
            )]
            forces_mae += [training_utils.mean_absolute_error(
                forces, inputs['forces'], msk=batch_segments_dict['node_mask']
            )]
            energy_mse += [training_utils.mean_squared_error(
                energy, inputs['energy'], msk=batch_segments_dict['graph_mask']
            )]
            forces_mse += [training_utils.mean_squared_error(
                forces, inputs['forces'], msk=batch_segments_dict['node_mask']
            )]
            if collect_predictions:
                energy_predictions += [energy]
                forces_predictions += [forces]
                energy_gt += [inputs['energy']]
                forces_gt += [inputs['forces']]
                graphs += [graph_batch]
            # eval_metrics_list += [eval_step_fn(params, inputs)]
            running_num_of_evaluated_structures += batch_segments_dict[
                'num_of_non_padded_graphs'
            ]
            if num_test is not None:
                if running_num_of_evaluated_structures >= num_test:
                    print(
                        f'Stop testing after {n=} batches since'
                        f' {running_num_of_evaluated_structures} test structures have'
                        ' been reached.'
                    )
                    break

        print(
            'Metrics are collected from'
            f' {running_num_of_evaluated_structures} test structures.'
        )
        # eval_metrics = collect_metrics(eval_metrics_list)
        energy_mse = jnp.array(energy_mse).mean()
        forces_mse = jnp.array(forces_mse).mean()
        energy_rmse = jnp.sqrt(energy_mse)
        forces_rmse = jnp.sqrt(forces_mse)
        energy_mae = jnp.array(energy_mae).mean()
        forces_mae = jnp.array(forces_mae).mean()

        metrics = {
            'energy_mse': energy_mse,
            'forces_mse': forces_mse,
            'energy_mae': energy_mae,
            'forces_mae': forces_mae,
            'energy_rmse': energy_rmse,
            'forces_rmse': forces_rmse,
        }

        metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)

        return metrics, (
            energy_predictions,
            forces_predictions,
            energy_gt,
            forces_gt,
            graphs
        )

    def run_training(
            self,
            model,
            optimizer,
            ckpt_dir=None,
            ckpt_manager_options=None,
            params=None,
            opt_state=None,
    ):
        """Run training for `model` given the `NpzTrainer` hyperparameters.

    Args:
      model:
      optimizer:
      ckpt_dir:
      ckpt_manager_options:
      params:
      opt_state:

    Returns:
    """
        if params is None:
            # initialize fresh parameters of the model
            params = model.init(
                jax.random.PRNGKey(self.model_seed),
                positions=jnp.zeros((self.max_num_nodes, 3)),
                atomic_numbers=jnp.zeros((self.max_num_nodes,), dtype=jnp.int16),
                atomic_dipoles=jnp.zeros((self.max_num_nodes, 3)),
                dst_idx=jnp.zeros((self.max_num_edges,), dtype=jnp.int32),
                src_idx=jnp.zeros((self.max_num_edges,), dtype=jnp.int32),
                batch_segments=jnp.zeros((self.max_num_nodes,), dtype=jnp.int32),
                graph_mask=jnp.array([True] * self.max_num_graphs),
                lattice_vectors=None if self.pbc_bool == False else jnp.zeros((self.max_num_graphs, 3, 3)),
                cell_offsets=None if self.pbc_bool == False else jnp.zeros((self.max_num_edges, 3))
            )
        else:
            # use params passed to the run method by creating a deep copy
            params = params.copy()

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        
        if self.use_wandb:
            wandb.log(
                data={'num_params': num_params}
            )
        else:
            print(
                f'Number of parameters: {num_params}'
            )

        # initialize the optimizer state
        if opt_state is None:
            # initialize a fresh optimizer state
            opt_state = optimizer.init(params)
        else:
            # use opt_state passed to the run method by creating deep copy
            opt_state = opt_state.copy()

        # create the loss function, for a given batch size and energy and force
        # weights
        loss_fn = training_utils.make_loss_fn(
            model,
            energy_weight=self.energy_weight,
            forces_weight=self.forces_weight,
        )

        # make the training and evaluation step function, and jit.compile them
        train_step_fn = jax.jit(
            training_utils.make_train_step_fn(loss_fn, optimizer)
        )
        eval_step_fn = jax.jit(training_utils.make_eval_step_fn(loss_fn))

        # initialize the dynamical batching of jraph.GraphTuple objects
        dynamically_batch = functools.partial(
            jraph.dynamically_batch,
            n_node=self.max_num_nodes,
            n_edge=self.max_num_edges,
            n_graph=self.max_num_graphs,
        )

        # create the function that takes care of the creation of the batch segments
        jitted_batch_segments_fn = jax.jit(jraph_utils.batch_segments_fn)

        # prepare the training and the validation data, that includes removing the
        # energy shift, converting units and creating the jraph.GraphTuples
        prepared_train_ds, prepared_valid_ds = (
            self.prepare_training_and_validation_data(cutoff=model.cutoff)
        )

        # make lists
        prepared_train_ds = list(prepared_train_ds)
        prepared_valid_ds = list(prepared_valid_ds)

        mngr = self.init_ckpt_manager(
            ckpt_dir=ckpt_dir, ckpt_manager_options=ckpt_manager_options
        )

        # check if a CheckpointManager already exists under the specified checkpoint
        # directory.
        init_step = 0
        if mngr is not None:
            if mngr.latest_step() is not None:
                print(f'Continue training from step {mngr.latest_step()}')
                ckpt_manager_state = mngr.restore(mngr.latest_step())
                params = ckpt_manager_state['params']
                opt_state = optimizer.init(params)
                init_step = mngr.latest_step()

        total_step = 0
        # running_processed_structures = jnp.array(0, dtype=jnp.int64)
        for _ in range(self.num_epochs):
            # at the beginning of each epoch shuffle the list of GraphTuples and
            # create a new iterator of batched graphs
            random.shuffle(prepared_train_ds)
            train_iter = dynamically_batch(prepared_train_ds)

            for graph_batch_train in train_iter:
                batch_segments_dict = jitted_batch_segments_fn(graph_batch_train)
                inputs = jraph_utils.jraph_to_input(graph_batch_train)
                # print(inputs['atomic_numbers'].shape)
                inputs.update(batch_segments_dict)
                params, opt_state, train_metrics = train_step_fn(
                    params, opt_state, inputs
                )

                train_metrics = train_metrics.compute()
                train_metrics = {
                    f'train_{k}': float(v) for k, v in train_metrics.items()
                }
                if (total_step + 1) % self.log_interval_steps == 0:
                    if self.use_wandb:
                        wandb.log(data=train_metrics, step=total_step)
                    else:
                        print(train_metrics)

                # Iterate over the full validation set
                if (init_step + total_step + 1) % self.save_interval_steps == 0:
                    # create a new iterator for validation
                    valid_iter = dynamically_batch(prepared_valid_ds)
                    eval_metrics: Any = None
                    for graph_batch_valid in valid_iter:
                        batch_segments_dict = jitted_batch_segments_fn(graph_batch_valid)
                        inputs = jraph_utils.jraph_to_input(graph_batch_valid)
                        inputs.update(batch_segments_dict)
                        eval_out = eval_step_fn(params, inputs)
                        eval_metrics = (
                            eval_out
                            if eval_metrics is None
                            else eval_metrics.merge(eval_out)
                        )

                    eval_metrics = eval_metrics.compute()
                    eval_metrics = {
                        f'eval_{k}': float(v) for k, v in eval_metrics.items()
                    }
                    if self.use_wandb:
                        wandb.log(data=eval_metrics, step=total_step)
                    else:
                        print('Eval metrics: ', eval_metrics)

                    if mngr is not None:
                        mngr.save(
                            init_step + total_step + 1,
                            {'params': params, 'opt_state': opt_state},
                            metrics={'loss': eval_metrics['eval_loss']},
                        )
                total_step += 1

        return params

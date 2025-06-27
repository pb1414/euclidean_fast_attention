"""Reference implementation for O(3) equivariant MPNN with and without EFA block."""

import e3x
import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping

from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence

from . import fast_attention
from .utils import space_utils

Array = jaxtyping.Array


class EnergyModel(nn.Module):
    """O3 equivariant MPNN for energy (and force) prediction. Use of EFA blocks can be optionally enabled / disabled.

    Attributes:
    cutoff: Cutoff used in message passing step.
    num_features: Number of features.
    num_layers: Number of layers.
    num_post_residual_mlps: Number of residual MLPs after adding the output of
      the local and the non local update.
    mp_max_degree: Maximal degree of during message passing (MP).
    mp_num_basis_fn: Number of basis functions for MP update.
    era_use_in_iterations: Iterations in which to use euclidean rope attention
      (ERA).
    era_max_degree: Maximal degree that is used in the ERA module.
    era_include_pseudotensors: Inlcude pseudotensors in ERA module.
    era_activation_fn: Which activation function to use in ERA module.
    era_num_frequencies: Number of frequencies to use in ERA. Defaults to
      `era_qk_num_features/2`.
    era_max_frequency: Maximal frequency in ERA.
    era_max_length: Maximal pairwise distance in the input. This ensures
      rotational invariance up to `era_max_length`.
    era_lebedev_num: Number of points in the Lebedev quadrature. To ensure
      rotational invariance, it must be chosen accordingly to
      `era_max_frequency`
    era_qk_num_features: Number of features for query/key vectors in ERA.
      Defaults to `num_features`.
    era_v_num_features: Number of features for value vectors in ERA. Defaults to
      `num_features`.
    output_is_zero_at_init: Output of the network is zero at initialization.
    use_switch: Use switching parameter to weight local and non local block.
    emulate_era_block: Emulate the ERA block, without actually using the
    core operation. This is, to get a local model with the same computational
    flow and just the core eucliddean rope attention mechanism removed.
    zmax: Maximal atomic number.
    """

    cutoff: float = 5.0
    cutoff_fn: str = 'smooth_cutoff'
    num_features: int = 32
    num_layers: int = 2
    num_post_residual_mlps: int = 0
    self_interaction: bool = False  # self interactions via iterated tensor products
    atomic_dipole_embedding: bool = False

    pbc_bool: bool = False

    mp_max_degree: int = 2
    radial_basis_fn: str = 'reciprocal_bernstein'
    mp_num_basis_fn: int = 32

    iterated_tensor_products: bool = False

    era_tensor_integration: bool = False
    era_ti_max_degree_sph: Optional[int] = None
    era_ti_max_degree: Optional[int] = None
    era_ti_parametrize_coupling_paths: bool = False
    era_ti_degree_scaling_constants: Optional[Sequence[float]] = None

    era_use_in_iterations: Optional[Sequence[int]] = None
    era_max_degree: Optional[int] = None
    era_include_pseudotensors: Optional[bool] = False

    era_activation_fn: Callable[..., Any] = lambda u: u

    era_num_frequencies: Optional[int] = None
    era_max_frequency: Optional[float] = None
    era_max_length: Optional[float] = None
    era_lebedev_num: Optional[int] = None

    era_qk_num_features: Optional[int] = None
    era_v_num_features: Optional[int] = None

    output_is_zero_at_init: Optional[bool] = True
    efa_block_behaves_like_identity_at_init: bool = True
    mp_block_behaves_like_identity_at_init: bool = True
    use_switch: bool = True

    emulate_era_block: bool = False

    zmax: int = 118

    def setup(self):
        if self.output_is_zero_at_init:
            self.last_layer_kernel_init = jax.nn.initializers.zeros
        else:
            self.last_layer_kernel_init = e3x.nn.modules.default_kernel_init

        if self.efa_block_behaves_like_identity_at_init:
            self.efa_last_layer_kernel_init = jax.nn.initializers.zeros
        else:
            self.efa_last_layer_kernel_init = e3x.nn.modules.default_kernel_init

        if self.mp_block_behaves_like_identity_at_init:
            self.mp_last_layer_kernel_init = jax.nn.initializers.zeros
        else:
            self.mp_last_layer_kernel_init = e3x.nn.modules.default_kernel_init

    def energy(
            self,
            atomic_numbers,
            positions,
            lattice_vectors,
            cell_offsets,
            atomic_dipoles,
            dst_idx,
            src_idx,
            batch_segments,
            graph_mask,
    ):
        num_nodes = len(atomic_numbers)
        num_graphs = len(graph_mask)

        # 1. Calculate displacement vectors.
        displacements = space_utils.calculate_displacement_vectors(
            positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            lattice_vectors=lattice_vectors,
            cell_offsets=cell_offsets,
            pbc_bool=self.pbc_bool
        )

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(
            displacements,
            num=self.mp_num_basis_fn,
            max_degree=self.mp_max_degree,
            radial_fn=getattr(e3x.nn, self.radial_basis_fn),
            cutoff_fn=functools.partial(
                getattr(e3x.nn, self.cutoff_fn),
                cutoff=self.cutoff
            ) if self.cutoff_fn is not None else None,
        )  # (num_pairs, 1, (max_degree+1)**2, num_basis_functions)

        # 3. Embed atomic numbers in feature space.
        x = e3x.nn.Embed(num_embeddings=self.zmax, features=self.num_features)(
            atomic_numbers
        )  # (N, 1, 1, num_features)

        # Use atomic dipoles to initialize the degree=1 features.
        if self.atomic_dipole_embedding:
            if atomic_dipoles is None:
                raise ValueError(
                    'atomic_dipoles must be passed to `energy_fn` for `atomic_dipole_embedding=True`.'
                )

            dipole_embeddings = nn.Dense(
                x.shape[-1],
                use_bias=False)(atomic_dipoles[:, None, :, None])  # (N, 1, 3, num_features)
            x = jnp.concatenate([x, dipole_embeddings], axis=2)  # (N, 1, 4, num_features)

        else:
            pass

        # ------------------------ begin of interaction blocks -------------------------------------------

        # Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_layers):
            # Message-pass.
            if i == self.num_layers - 1:  # Final iteration.

                # Since we will only use scalar features after the final message-pass,
                # we do not want to produce non-scalar features for efficiency reasons.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx, num_segments=num_nodes
                )

                # After the final updates, we can safely throw away all non-scalar features.
                if not self.atomic_dipole_embedding:
                    x = e3x.nn.change_max_degree_or_type(
                        x, max_degree=0, include_pseudotensors=False
                    )

                # skip connection around mp block
                y = e3x.nn.add(x, y)

                # Atom-wise refinement MLP for message passing features.
                y = e3x.nn.Dense(self.num_features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(
                    self.num_features, kernel_init=self.mp_last_layer_kernel_init
                )(y)

                # Apply non local interactions via EFA block.
                if self.era_use_in_iterations is not None:

                    # Check if the current layer is one of the layers that should use EFA.
                    if i in self.era_use_in_iterations:
                        # Check if the EFA block should be emulated.
                        if self.emulate_era_block:
                            # Emulation of the EFA block via a Dense layer.
                            # Atom-wise refinement MLP for non local features.
                            
                            y_nl = e3x.nn.Dense(
                                2 * self.era_qk_num_features + self.era_v_num_features
                                )(
                                    e3x.nn.change_max_degree_or_type(
                                        x,
                                        max_degree=0,
                                        include_pseudotensors=False,
                                    )
                                )
                        else:
                            # Use the EFA block.
                            y_nl = fast_attention.EuclideanFastAttention(
                                pbc_bool=self.pbc_bool,
                                lebedev_num=self.era_lebedev_num,
                                num_features_qk=self.era_qk_num_features,
                                num_features_v=self.era_v_num_features,
                                activation_fn=self.era_activation_fn,
                                epe_num_frequencies=self.era_num_frequencies,
                                epe_max_frequency=self.era_max_frequency,
                                epe_max_length=self.era_max_length,
                                tensor_integration=self.era_tensor_integration,
                                ti_max_degree_sph=self.era_ti_max_degree_sph,
                                ti_max_degree=self.era_ti_max_degree,
                                ti_parametrize_coupling_paths=self.era_ti_parametrize_coupling_paths,
                                ti_degree_scaling_constants=self.era_ti_degree_scaling_constants,
                                name=f'EuclideanRopeAttention_{i}'  # for backward compatibility with old module name
                            )(
                                e3x.nn.change_max_degree_or_type(
                                    x,
                                    max_degree=0,
                                    include_pseudotensors=False,
                                ),
                                positions,
                                batch_segments,
                                graph_mask,
                                lattice_vectors=lattice_vectors,
                            )

                        # Skip connection around EFA.
                        y_nl = e3x.nn.add(e3x.nn.Dense(self.num_features)(y_nl), x)

                        # Atom-wise refinement MLP for non local features.
                        y_nl = e3x.nn.Dense(self.num_features)(y_nl)
                        y_nl = e3x.nn.silu(y_nl)
                        y_nl = e3x.nn.Dense(
                            self.num_features, 
                            kernel_init=self.efa_last_layer_kernel_init
                        )(y_nl)

                    else:
                        y_nl = jnp.zeros_like(y)
                else:
                    y_nl = jnp.zeros_like(y)

            else:
                # In intermediate iterations, the message-pass should consider all possible coupling paths.
                y = e3x.nn.MessagePass()(
                    x,
                    basis,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                    num_segments=num_nodes,
                )

                # skip connection around mp block
                y = e3x.nn.add(x, y)

                # Atom-wise refinement MLP for message passing features.
                y = e3x.nn.Dense(self.num_features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(
                    self.num_features, kernel_init=self.mp_last_layer_kernel_init
                )(y)

                # Apply non local interactions using Euclidean RoPE.
                if self.era_use_in_iterations is not None:
                    if i in self.era_use_in_iterations:
                        if self.emulate_era_block:
                            # Emulation of the EFA block via a Dense layer.
                            # Atom-wise refinement MLP for non local features.
                            y_nl = e3x.nn.Dense(
                                2 * self.era_qk_num_features + self.era_v_num_features
                                )(
                                    x
                                )
                        else:
                            y_nl = fast_attention.EuclideanFastAttention(
                                pbc_bool=self.pbc_bool,
                                lebedev_num=self.era_lebedev_num,
                                num_features_qk=self.era_qk_num_features,
                                num_features_v=self.era_v_num_features,
                                activation_fn=self.era_activation_fn,
                                epe_num_frequencies=self.era_num_frequencies,
                                epe_max_frequency=self.era_max_frequency,
                                epe_max_length=self.era_max_length,
                                tensor_integration=self.era_tensor_integration,
                                ti_max_degree_sph=self.era_ti_max_degree_sph,
                                ti_max_degree=self.era_ti_max_degree,
                                ti_parametrize_coupling_paths=self.era_ti_parametrize_coupling_paths,
                                ti_degree_scaling_constants=self.era_ti_degree_scaling_constants,
                                name=f'EuclideanRopeAttention_{i}'
                            )(
                                e3x.nn.change_max_degree_or_type(
                                    x,
                                    max_degree=self.era_max_degree if x.shape[-2] > 1 else 0,
                                    include_pseudotensors=self.era_include_pseudotensors,
                                ),
                                positions,
                                batch_segments,
                                graph_mask,
                                lattice_vectors=lattice_vectors,
                            )

                        # skip connection around RoPe attention
                        y_nl = e3x.nn.add(e3x.nn.Dense(self.num_features)(y_nl), x)

                        # Atom-wise refinement MLP for non local features.
                        y_nl = e3x.nn.Dense(self.num_features)(y_nl)
                        y_nl = e3x.nn.silu(y_nl)
                        y_nl = e3x.nn.Dense(
                            self.num_features, kernel_init=self.efa_last_layer_kernel_init
                        )(y_nl)

                    else:
                        y_nl = jnp.zeros_like(y)
                else:
                    y_nl = jnp.zeros_like(y)

            # --------------- Post local and non local block --------------------------------------

            # Use a trainable switching value between local (from MP) and non-local (from EFA) features.
            if self.use_switch:
                c_switch = self.param(
                    f'c_switch_layer_{i}',
                    jax.nn.initializers.constant(jnp.array(0.5)),
                    (1,),
                )

                # Residual connection.
                x = e3x.nn.add(
                    e3x.nn.add(x, jax.nn.silu(c_switch) * y),
                    jax.nn.silu(1 - c_switch) * y_nl,
                )

                # To couple local and global directional information, i.e. when local vectorial embeddings
                # such like atomic dipoles are present we perform a self-interaction CG tensor contraction per atom.
                if self.atomic_dipole_embedding or self.self_interaction:
                    if i == self.num_layers - 1:
                        # Let local representations interact with the global representations via self-interaction.
                        z = e3x.nn.TensorDense(
                            include_pseudotensors=False,
                            max_degree=0
                        )(x)
                    else:
                        z = e3x.nn.TensorDense()(x)

                    x = e3x.nn.change_max_degree_or_type(
                        x,
                        max_degree=0,
                        include_pseudotensors=False
                    )
                    x = e3x.nn.add(x, z)
            else:
                # Residual connection.
                x = e3x.nn.add(e3x.nn.add(x, y), y_nl)

                if self.atomic_dipole_embedding or self.self_interaction:
                    if i == self.num_layers - 1:
                        # Let local representations interact with the global representations via self-interaction.
                        z = e3x.nn.TensorDense(
                            include_pseudotensors=False,
                            max_degree=0
                        )(x)
                    else:
                        z = e3x.nn.TensorDense()(x)

                    x = e3x.nn.change_max_degree_or_type(
                        x,
                        max_degree=0,
                        include_pseudotensors=False
                    )
                    x = e3x.nn.add(x, z)

            # Apply post residual MLPs.
            for _ in range(self.num_post_residual_mlps):
                y = e3x.nn.Dense(self.num_features)(x)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(
                    self.num_features, kernel_init=self.last_layer_kernel_init
                )(y)
                x = e3x.nn.add(x, y)

        # --------------- End of interaction blocks ---------------------------------------------

        # Predict atomic energies with an ordinary dense layer.
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.zmax + 1),
        )

        # Only keep the invariant degree without pseudotensors.
        x = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )  # (..., num_atoms, 1, 1, num_features)

        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=self.last_layer_kernel_init
        )(
            x
        )  # (..., num_atoms, 1, 1, num_features)

        atomic_energies = jnp.squeeze(
            atomic_energies, axis=(-1, -2, -3)
        )  # Squeeze last 3 dimensions.

        atomic_energies += jnp.take(element_bias, atomic_numbers)

        # Sum atomic energies to obtain the total energy.
        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=num_graphs
        )

        # For padded graphs set energies to zero.
        energy = jnp.where(graph_mask, energy, 0)

        # To be able to efficiently compute forces, our model should return a single
        # output (instead of one for each molecule in the batch). Fortunately, since
        # all atomic contributions only influence the energy in their own batch
        # segment, we can simply sum the energy of all molecules in the batch to
        # obtain a single proxy output to differentiate.
        return (
            -jnp.sum(energy),
            energy,
        )  # Forces are the negative gradient, hence the minus sign.

    @nn.compact
    def __call__(
            self,
            atomic_numbers,
            positions,
            dst_idx,
            src_idx,
            batch_segments: Optional[Array] = None,
            graph_mask: Optional[Array] = None,
            lattice_vectors: Optional[Array] = None,
            cell_offsets: Optional[Array] = None,
            atomic_dipoles: Optional[Array] = None,
            calculate_forces: bool = True
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            graph_mask = jnp.array([True])
        if calculate_forces:
            # Since we want to also predict forces, i.e. the gradient of the energy
            # w.r.t. positions (argument 1), we use jax.value_and_grad to create a
            # function for predicting both energy and forces for us.
            energy_and_forces = jax.value_and_grad(
                self.energy, argnums=1, has_aux=True
            )
            (_, energy), forces = energy_and_forces(
                atomic_numbers,
                positions,
                lattice_vectors,
                cell_offsets,
                atomic_dipoles,
                dst_idx,
                src_idx,
                batch_segments,
                graph_mask,
            )

            return energy, forces
        else:
            return self.energy(
                atomic_numbers,
                positions,
                lattice_vectors,
                cell_offsets,
                atomic_dipoles,
                dst_idx,
                src_idx,
                batch_segments,
                graph_mask,
            )[1]

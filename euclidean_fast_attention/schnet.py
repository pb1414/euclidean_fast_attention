import flax.linen as nn
import jax
import jax.numpy as jnp
import e3x
from euclidean_fast_attention.fast_attention import EuclideanFastAttention
from typing import Callable, Optional
from jaxtyping import Array

from .utils import space_utils


def shifted_softplus(x):
    return jax.nn.softplus(x) + jnp.log(jnp.asarray(0.5, dtype=x.dtype))


class CfConv(nn.Module):
    @nn.compact
    def __call__(
            self,
            x,  # (num_nodes, num_features)
            rbf,  # (num_pairs, num_rbf)
            cut,  # (num_pairs)
            dst_idx,  # (num_pairs)
            src_idx,  # (num_pairs)
    ):

        num_features = x.shape[-1]
        num_nodes = len(x)

        W = shifted_softplus(nn.Dense(num_features)(shifted_softplus(nn.Dense(num_features)(rbf))))
        # (num_pairs, num_features)

        W = W * jnp.expand_dims(cut, axis=-1)  # (num_pairs, num_features)

        y = jax.ops.segment_sum(
            x[src_idx] * W,
            segment_ids=dst_idx,
            num_segments=num_nodes
        )  # (num_pairs, num_features)

        return y


class Interaction(nn.Module):
    @nn.compact
    def __call__(
            self,
            x,  # (num_nodes, num_features)
            rbf,  # (num_pairs, num_rbf)
            cut,  # (num_pairs)
            dst_idx,  # (num_pairs)
            src_idx,  # (num_pairs)
    ):
        num_features = x.shape[-1]

        y = nn.Dense(num_features)(x)  # (num_nodes, num_features)
        y = CfConv()(
            x=y,
            rbf=rbf,
            cut=cut,
            dst_idx=dst_idx,
            src_idx=src_idx
        )  # (num_nodes, num_features)
        y = nn.Dense(num_features)(shifted_softplus(nn.Dense(num_features)(y)))  # (num_nodes, num_features)
        return y


class EFABlock(nn.Module):
    era_max_length: float
    pbc_bool: bool
    mlp_hidden_features: Optional[int] = None
    layer_normalization_bool: bool = False
    era_emulate_bool: bool = False
    era_lebedev_num: int = 50
    era_max_frequency: float = jnp.pi
    era_qk_num_features: int = 16
    era_v_num_features: int = 32
    era_activation_fn: Callable = e3x.nn.gelu
    behaves_like_identity_at_init: bool = True
    
    def setup(self):
        if self.behaves_like_identity_at_init == True:
            self.last_layer_kernel_init_fn = jax.nn.initializers.zeros            
        else:
            self.last_layer_kernel_init_fn = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(
            self,
            x,
            positions,
            batch_segments,
            graph_mask,
            lattice_vectors,
    ):

        num_features = x.shape[-1]
        if self.era_emulate_bool == True:
            y_att = e3x.nn.Dense(
                2 * self.era_qk_num_features + self.era_v_num_features
            )(
                x
            )
        else:
            y_att = EuclideanFastAttention(
                num_features_qk=self.era_qk_num_features,
                num_features_v=self.era_v_num_features,
                activation_fn=self.era_activation_fn,
                lebedev_num=self.era_lebedev_num,
                epe_max_frequency=self.era_max_frequency,
                epe_max_length=self.era_max_length,
                pbc_bool=self.pbc_bool,
            )(
                nn.LayerNorm()(x) if self.layer_normalization_bool == True else x,
                positions,
                batch_segments,
                graph_mask,
                lattice_vectors=lattice_vectors
            )

        y_att = e3x.nn.Dense(
            num_features, 
            kernel_init=self.last_layer_kernel_init_fn
        )(y_att)

        # Skip around EFA block.
        x = e3x.nn.add(x, y_att)

        # Atom-wise refinement MLP for non local features.
        if self.mlp_hidden_features is not None:
            mlp_hidden_features = self.mlp_hidden_features
        else:
            mlp_hidden_features = num_features

        y_mlp = e3x.nn.Dense(mlp_hidden_features)(nn.LayerNorm()(x) if self.layer_normalization_bool == True else x)
        y_mlp = e3x.nn.silu(y_mlp)
        y_mlp = e3x.nn.Dense(num_features, kernel_init=self.last_layer_kernel_init_fn)(y_mlp)

        # Skip around MLP.
        x = e3x.nn.add(x, y_mlp)

        return x


class SchNet(nn.Module):
    num_layers: int = 3
    num_features: int = 128

    cutoff: float = 5.

    pbc_bool: bool = False

    radial_basis_fn: str = 'reciprocal_bernstein'
    num_basis_fn: int = 32

    zmax: int = 119

    use_efa_block: bool = False
    emulate_efa_block: bool = False
    
    efa_block_behaves_like_identity_at_init: bool = True
    efa_block_layer_normalization_bool: bool = False
    efa_block_mlp_hidden_features: Optional[int] = None
    era_lebedev_num: Optional[int] = None
    era_max_frequency: Optional[float] = None
    era_max_length: Optional[float] = None
    era_qk_num_features: Optional[int] = None
    era_v_num_features: Optional[int] = None
    era_activation_fn: Optional[Callable] = None

    def setup(self):
        if self.use_efa_block:
            try:
                assert self.era_lebedev_num is not None
                assert self.era_max_length is not None
                assert self.era_qk_num_features is not None
                assert self.era_v_num_features is not None
                assert self.era_max_frequency is not None
                assert self.efa_block_layer_normalization_bool is not None
                assert self.era_activation_fn is not None
            except AssertionError:
                raise ValueError(
                    "If use_efa_block is True, all EFA block parameters must be specified."
                    "Received: "
                    f"era_lebedev_num={self.era_lebedev_num}, "
                    f"era_max_length={self.era_max_length}, "
                    f"era_qk_num_features={self.era_qk_num_features}, "
                    f"era_v_num_features={self.era_v_num_features}, "
                    f"efa_block_layer_normlization_bool={self.efa_block_layer_normlization_bool}"
                    f"era_activation_fn={self.era_activation_fn}"
                    f"era_max_frequency={self.era_max_frequency}"
                )
        if self.emulate_efa_block == True:
            if self.use_efa_block == False:
                raise ValueError(
                    f"If EFA block should be emulated it must be activated. "
                    f"received {self.emulate_efa_block=} but {self.use_efa_block=}."
                )

    def energy(
            self,
            atomic_numbers,
            positions,
            lattice_vectors,
            cell_offsets,
            dst_idx,
            src_idx,
            batch_segments,
            graph_mask,
    ):
        num_nodes = len(atomic_numbers)
        num_graphs = len(graph_mask)

        displacements = space_utils.calculate_displacement_vectors(
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            lattice_vectors=lattice_vectors,
            cell_offsets=cell_offsets,
            pbc_bool=self.pbc_bool
        )  # (num_pairs, 3)

        # Calculate distances.
        distances = e3x.ops.norm(displacements, axis=-1, keepdims=True)  # (num_pairs, 1)
        # Squeeze last axis in distances.
        distances = distances.squeeze(-1)  # (num_pairs)

        # Calculate radial basis functions and cut.
        rbf = getattr(
            e3x.nn,
            self.radial_basis_fn
        )(
            distances,
            num=self.num_basis_fn
        )  # (num_pairs, num_rbf)

        cut = e3x.nn.smooth_cutoff(
            distances,
            cutoff=self.cutoff
        )  # (num_pairs)

        # Embed atomic numbers in feature space.
        x = nn.Embed(num_embeddings=self.zmax, features=self.num_features)(
            atomic_numbers
        )  # (N, num_features)

        # Iterate MP steps.
        for i in range(self.num_layers):
            
            # Local interaction
            delta_x = Interaction()(
                x=x,
                rbf=rbf,
                cut=cut,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )  # (num_nodes, num_features)

            # EFA block.
            if self.use_efa_block:
                x_nl = x[:, None, None]
                x_nl = EFABlock(
                    era_lebedev_num=self.era_lebedev_num,
                    era_max_frequency=self.era_max_frequency,
                    era_max_length=self.era_max_length,
                    era_qk_num_features=self.era_qk_num_features,
                    era_v_num_features=self.era_v_num_features,
                    era_emulate_bool=self.emulate_efa_block,
                    behaves_like_identity_at_init=self.efa_block_behaves_like_identity_at_init,
                    pbc_bool=self.pbc_bool,
                    mlp_hidden_features=self.efa_block_mlp_hidden_features,
                    layer_normalization_bool=self.efa_block_layer_normalization_bool,
                    era_activation_fn=self.era_activation_fn
                )(
                    x=x_nl,
                    positions=positions,
                    batch_segments=batch_segments,
                    graph_mask=graph_mask,
                    lattice_vectors=lattice_vectors,
                )  # (num_nodes, 1, 1, num_features)

                x_nl = jnp.squeeze(x_nl, axis=(-2, -3))  # (num_nodes, num_features)
            else:
                x_nl = jnp.zeros_like(x)

            # Skip around MP and EFA block.
            x = x + delta_x + x_nl

        num_features = x.shape[-1]

        # Predict atomic energies with an MLP.
        atomic_energies = nn.Dense(1)(shifted_softplus(nn.Dense(num_features // 2)(x)))  # (num_nodes, 1)

        atomic_energies = jnp.squeeze(
            atomic_energies, axis=(-1)
        )  # (num_nodes)

        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.zmax + 1),
        )
        atomic_energies += jnp.take(element_bias, atomic_numbers)

        # 6. Sum atomic energies to obtain the total energy.
        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=num_graphs
        )

        # For padded graphs set energies to zero.
        energy = jnp.where(graph_mask, energy, 0)

        return (
            -jnp.sum(energy),
            energy
        )

    @nn.compact
    def __call__(
            self,
            atomic_numbers,
            positions,
            dst_idx,
            src_idx,
            batch_segments=None,
            graph_mask=None,
            lattice_vectors: Optional[Array] = None,
            cell_offsets: Optional[Array] = None,
            atomic_dipoles=None,
            calculate_forces=True
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
                dst_idx,
                src_idx,
                batch_segments,
                graph_mask,
            )[1]

import flax.linen as nn
import jax
import jax.numpy as jnp
import e3x
from e3x.nn.functions import smooth_switch
from euclidean_fast_attention.utils.space_utils import calculate_displacement_vectors


def smooth_long_range(dij, pij, x_on, x_off, pol_decay):

    def short_range_fn(_dij):
        return 1.0 / jnp.power(1 + _dij, pol_decay)

    def long_range_fn(_dij):
        return 1.0 / (jnp.power(_dij, pol_decay) + 1e-5)


    return pij * (short_range_fn(dij) * (1 - smooth_switch(dij, x0=x_on, x1=x_off)) + long_range_fn(dij) * smooth_switch(dij, x0=x_on, x1=x_off))


class DispersionEnergy(nn.Module):
    model_cutoff: float
    pbc_bool: bool = False
    double_counting: bool = True

    def setup(self):
        if self.pbc_bool == True:
            raise NotImplementedError("PBCs are not implemented for Dipsersion module yet.")
        if self.double_counting == True:
            self.pre_factor = 0.5
        else:
            self.pre_factor = 1.0
        
    
    @nn.compact
    def __call__(
        self,
        x,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments, 
        graph_mask,
        lattice_vectors=None,
        cell_offsets=None
    ):
        """
        Computes the dispersion energy of a system. PBCs are not implemented yet.
        
        Args:
            x: (num_atoms, 1 or 2, (max_degree+1)**2, num_features) - features of the atoms
            atomic_numbers: (num_atoms, ) - atomic numbers of the atoms
            positions: (num_atoms, 3) - positions of the atoms
            dst_idx: (num_pairs, ) - indices of the destination atoms
            src_idx: (num_pairs, ) - indices of the source atoms
            batch_segments: (num_atoms, ) - segments of the batch
            graph_mask: (num_atoms, ) - mask for the graph
            lattice_vectors: (3, 3) - lattice vectors
            cell_offsets: (num_pairs, 3) - offsets of the atoms in the unit cell
        """
        
        del lattice_vectors
        del cell_offsets

        num_features = x.shape[-1]

        rij = calculate_displacement_vectors(
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            lattice_vectors=None,
            cell_offsets=None,
            pbc_bool=self.pbc_bool,
        )  # (num_pairs, 3)

        dij = e3x.ops.norm(rij, axis=-1)  # (num_pairs, )

        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)  # (num_atoms, 1, 1, num_features)

        c6coeff = e3x.nn.Dense(features=1)(e3x.nn.silu(e3x.nn.Dense(features=num_features)(x)))  # (num_atoms, 1, 1, 1)
        c6coeff = jnp.squeeze(c6coeff, axis=(1, 2, 3))  # (num_atoms, )

        c6_bias = self.param("c6_bias", nn.initializers.zeros, (119, ))
        c6_bias = jnp.take(c6_bias, atomic_numbers)  # (num_atoms, )

        # c6coeff = c6_bias  # (num_atoms, )
        c6coeff = jnp.square(c6coeff + c6_bias)   # (num_atoms, )
        c6coeff_src = c6coeff[src_idx]  # (num_pairs, )
        c6coeff_dst = c6coeff[dst_idx]  # (num_pairs, )

        dispersion_energy_per_edge = -1.0 * smooth_long_range(
            dij,
            c6coeff_src * c6coeff_dst,
            x_on=1/4*self.model_cutoff,
            x_off=3/4*self.model_cutoff,
            pol_decay=6
        )  # (num_pairs, )
        
        dispersion_energy_per_atom = jax.ops.segment_sum(
            dispersion_energy_per_edge,
            segment_ids=dst_idx,
            num_segments=len(positions)
        )  # (num_atoms, )

        dispersion_energy_per_graph = jax.ops.segment_sum(
            dispersion_energy_per_atom,
            segment_ids=batch_segments,
            num_segments=len(graph_mask)
        )  # (num_graphs, )

        dispersion_energy_per_graph = self.pre_factor * jnp.where(graph_mask, dispersion_energy_per_graph, 0.0)  # (num_graphs, )

        return dispersion_energy_per_graph

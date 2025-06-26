import e3x
import jax.numpy as jnp
from typing import Optional
from jaxtyping import Array


def add_cell_offsets(r_ij: Array, lattice_vectors: Array, cell_offsets: Array) -> Array:
    """
    Add offsets to distance vectors given a cell and cell offsets. Cell vectors are assumed to be row-wise.
    Args:
        r_ij (Array): Distance vectors, shape: (num_pairs, 3)
        lattice_vectors (Array): lattice vectors, shape: (num_pairs, 3, 3). Lattice vectors are assumed to be row-wise following convention of ASE.
        cell_offsets (Array): Offsets for each pairwise distance, shape: (num_pairs, 3).
    Returns:
        Array: Adjusted distance vectors with offsets applied, shape: (num_pairs, 3).
    """

    offsets = jnp.einsum('Pi, Pij -> Pj', cell_offsets, lattice_vectors)
    return r_ij + offsets


def calculate_displacement_vectors(
        positions, 
        dst_idx, 
        src_idx,
        batch_segments,
        lattice_vectors: Optional[Array] = None, 
        cell_offsets: Optional[Array] = None, 
        pbc_bool: bool = False
    ) -> Array:
    """
    Calculate displacement vectors between atoms, optionally applying periodic boundary conditions.
    Args:
        positions (Array): Atom positions, shape: (num_atoms, 3).
        dst_idx (Array): Indices of destination atoms, shape: (num_pairs,).
        src_idx (Array): Indices of source atoms, shape: (num_pairs,).
        batch_segments (Array): Segment indices for batching, shape: (num_nodes,).
        lattice_vectors (Optional[Array]): Lattice vectors for periodic boundary conditions, shape: (num_graphs, 3, 3).
        cell_offsets (Optional[Array]): Offsets for each pairwise distance, shape: (num_pairs, 3).
        pbc_bool (bool): Whether to apply periodic boundary conditions. If True, `lattice_vectors` and `cell_offsets` must be provided.
    """
    
    """Calculate displacement vectors between atoms."""
    # Gather positions of destination and source atoms.
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    
    # Calculate displacement vectors. 
    # dst_idx denote the central atoms and src_idx the neighbor atoms. For non-periodic systems 
    # this does not matter but for periodic systems dst_idx should be the central atoms and src_idx the neighbor atoms 
    # since the add_cell_offsets function expects the displacements to point from from the central atom to the neighbor atom. 
    # i.e. pos_dst + displacements = pos_src
    displacements = positions_src - positions_dst
    if pbc_bool is True:    
        if lattice_vectors is None:
            raise ValueError(
                'lattice_vectors must be passed to `energy_fn` for `pbc_bool=True`.'
            )
        if cell_offsets is None:
            raise ValueError(
                'cell_offsets must be passed to `energy_fn` for `pbc_bool=True`.'
            )
        
        lattice_vectors = lattice_vectors[batch_segments] # Lattice vectors for each node.
        lattice_vectors = lattice_vectors[dst_idx] # Lattice vectors for each pair / edge.
        
        # Apply periodic boundary conditions.
        displacements = add_cell_offsets(
            displacements, lattice_vectors=lattice_vectors, cell_offsets=cell_offsets
        )
        return displacements
    else:
        return displacements
"""Utility functions for using jraph in the input pipline."""
import jax.numpy as jnp
import jaxtyping
import jraph
import numpy as np

from ase import neighborlist as ase_neighborlist
from typing import Any


Float = jaxtyping.Float
Array = jaxtyping.Array
Integer = jaxtyping.Integer


def compute_senders_and_receivers_np(
    positions, cutoff: float
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(np.int32)
    receivers = keep_edges[1].astype(np.int32)
    return senders, receivers

def compute_senders_and_receivers_pbc(
    positions,
    lattice_vectors,
    cutoff: float,
):
    
    senders, receivers, cell_offsets = ase_neighborlist.primitive_neighbor_list(
        'ijS',
        pbc=np.array(
            [True, True, True]
        ),
        cell=lattice_vectors,
        positions=positions,
        cutoff=cutoff,
        self_interaction=False
    )
    
    return senders, receivers, cell_offsets


def create_graph_tuple(
        element: dict[str, Any],
        cutoff: float,
        pbc_bool: bool
) -> jraph.GraphsTuple:
    """Takes a data element and wraps relevant components in a GraphsTuple."""
    atomic_numbers = element['atomic_numbers']
    positions = element['coordinates']
    energy = element['energy']
    forces = element['forces']
    atomic_dipoles = element.get('atomic_dipoles')
    node_mask = element['node_mask']
    if node_mask is None:
        node_mask = np.ones((len(atomic_numbers), )).astype(bool)
    
    if pbc_bool == False:
        senders, receivers = compute_senders_and_receivers_np(positions[node_mask], cutoff)
        lattice_vectors = None
        cell_offsets = None
    else:
        lattice_vectors = element.get('lattice_vectors', None)
        if lattice_vectors is None:
            raise ValueError(
                f'lattice_vectors must be provided when running with PBCs. received {lattice_vectors=}.'
            )
        senders, receivers, cell_offsets = compute_senders_and_receivers_pbc(
            positions=positions[node_mask], 
            lattice_vectors=lattice_vectors, 
            cutoff=cutoff
        )
        lattice_vectors = lattice_vectors.reshape(1, 3, 3)

    num_nodes = np.sum(node_mask)
    num_edges = len(receivers)
    return jraph.GraphsTuple(
        n_node=np.array([num_nodes]).astype(int),
        n_edge=np.array([num_edges]),
        senders=senders,
        receivers=receivers,
        nodes={
            'atomic_numbers': atomic_numbers[node_mask],
            'positions': positions[node_mask],
            'forces': forces[node_mask],
            'atomic_dipoles': atomic_dipoles[node_mask] if atomic_dipoles is not None else None
        },
        globals={
            'lattice_vectors': lattice_vectors,
            'energy': energy.reshape(-1),
        },
        edges={
            'cell_offsets': cell_offsets,
        },
    )


def jraph_to_input(x: jraph.GraphsTuple):
    atomic_numbers = x.nodes['atomic_numbers']
    positions = x.nodes['positions']

    # note the opposite convention compared to jraph
    src_idx = x.receivers
    dst_idx = x.senders
    energy = x.globals['energy']
    lattice_vectors = x.globals['lattice_vectors']
    cell_offsets = x.edges['cell_offsets']
    forces = x.nodes['forces']
    atomic_dipoles = x.nodes.get('atomic_dipoles')

    return {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'atomic_dipoles': atomic_dipoles,
        'energy': energy,
        'forces': forces,
        'src_idx': src_idx,
        'dst_idx': dst_idx,
        'lattice_vectors': lattice_vectors,
        'cell_offsets': cell_offsets
    }


def batch_segments_fn(x):
    """Function that creates batch segments from a batched jraph.GraphsTuple.

    Args:
        x: jraph.GraphsTuple.

    Returns:

    """
    
    num_graphs = len(x.n_node)
    num_nodes = x.nodes['atomic_numbers'].shape[0]

    batch_segments = jnp.repeat(
        jnp.arange(num_graphs), x.n_node, total_repeat_length=num_nodes
    )

    graph_mask = jraph.get_graph_padding_mask(x)
    node_mask = jraph.get_node_padding_mask(x)
    edge_mask = jraph.get_edge_padding_mask(x)
    num_of_padded_graphs = jraph.get_number_of_padding_with_graphs_graphs(x)

    return {
        'batch_segments': batch_segments,
        'graph_mask': graph_mask,
        'node_mask': node_mask,
        'edge_mask': edge_mask,
        'num_of_non_padded_graphs': num_graphs - num_of_padded_graphs,
    }

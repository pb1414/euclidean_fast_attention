import jraph
import jax
import numpy as np
import sys


from absl import app, flags
from euclidean_fast_attention.utils.nacl_toy import place_atoms_in_sphere
from euclidean_fast_attention.utils.nacl_toy import NaClPotential
from euclidean_fast_attention.utils.nacl_toy import sparse_pairwise_indices_np
from euclidean_fast_attention.utils.jraph_utils import batch_segments_fn

from pathlib import Path


FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', None, 'Save directory for the data.')
flags.DEFINE_string('filename', None, 'Save data to ${filename}.npz.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('num_data', 10_000, 'Number of data points')
flags.DEFINE_integer('Nmin', 2, 'Minimal number of atoms.')
flags.DEFINE_integer('Nmax', 100, 'Maximal number of atoms.')
flags.DEFINE_float('Dsphere', 15.0, 'Diameter of the sphere.')
flags.DEFINE_bool('repulsion_bool', True, 'add repulsion term.')


def main(_):
    
    if FLAGS.save_dir is None:
        print("Error: --save_dir must be specified.", file=sys.stderr)
        sys.exit(1)
    
    if FLAGS.seed is None:
        print("Error: --seed must be specified.", file=sys.stderr)
        sys.exit(1)
    
    # Diameter of the sphere.        
    D_sphere = FLAGS.Dsphere

    # Maximal number of atoms in the sphere.
    Nmax = FLAGS.Nmax
    Nmin = FLAGS.Nmin

    # Seed and number of data points.
    seed = FLAGS.seed
    num_data = FLAGS.num_data

    # Save directory.
    save_dir = Path(FLAGS.save_dir).resolve()
    save_dir.mkdir(exist_ok=True)

    # Potential details.
    repulsion_bool = FLAGS.repulsion_bool

    # Filename.
    filename = FLAGS.filename

    # Seed.
    np.random.seed(seed)

    nacl_potential = NaClPotential.create(
        pbc_bool=False, 
        repulsion_bool=repulsion_bool
    )

    @jax.jit
    def NaCl_energy_and_force_fn(positions, atomic_numbers, charges, src_idx, dst_idx, batch_segments, graph_mask):
        def energy_fn(pos):
            return nacl_potential.real_space(
                pos,
                atomic_numbers=atomic_numbers,
                charges=charges,
                src_idx=src_idx,
                dst_idx=dst_idx,
                batch_segments=batch_segments,
                graph_mask=graph_mask
            )
        (_, energy), forces = jax.value_and_grad(energy_fn, has_aux=True)(
                positions
            )

        return energy, forces

    # Sample number of atoms.
    N = np.random.randint(low=Nmin, high=Nmax + 1, size=(num_data, ))

    all_graphs = []

    # Create the graphs.
    print('Generate the geometries')
    for step, n in enumerate(N):

        z, pos = place_atoms_in_sphere(D_sphere, n)
        
        src_idx, dst_idx = sparse_pairwise_indices_np(n)
        
        all_graphs.append(
            jraph.GraphsTuple(
                globals=dict(),
                nodes=dict(
                    positions=pos,
                    atomic_numbers=z,
                    charges=np.where(z < 13, 1.0, -1.0)
                ),
                edges=dict(),
                senders=src_idx,
                receivers=dst_idx,
                n_node=np.array([n]),
                n_edge=np.array([len(src_idx)])
            )
        )

        if (step + 1) % np.floor(N.shape[-1] * 0.1).item() == 0:
            print(f'Step: {step + 1}')


    data_graphs = []

    step = 0
    print('Calculate energy and forces')
    for g in jraph.dynamically_batch(
        all_graphs,
        n_node=Nmax + 1,
        n_edge=Nmax * Nmax + 1,
        n_graph=2,
    ):

        batch_info = batch_segments_fn(g)

        energy, forces = NaCl_energy_and_force_fn(
            positions=g.nodes['positions'],
            atomic_numbers=g.nodes['atomic_numbers'],
            charges=g.nodes['charges'],
            src_idx=g.senders,
            dst_idx=g.receivers,
            graph_mask=batch_info['graph_mask'],
            batch_segments=batch_info['batch_segments']
        )
        g.globals['energy'] = np.array(energy)
        g.nodes['forces'] = np.array(forces)

        og_graphs = jraph.unbatch(g)[:batch_info['num_of_non_padded_graphs']]

        data_graphs += og_graphs
        step += 1

        if step % 10 == 0:
            print('Step: ', step)

    del all_graphs
    
    all_positions = []
    all_numbers = []
    all_energies = []
    all_forces = []
    all_nodes_masks = []
    # Pad the graphs and append the padded arrays.
    print('Prepare data for saving.')
    for dg in data_graphs:
        f = dg.nodes['forces']
        p = dg.nodes['positions']
        e = dg.globals['energy']
        z = dg.nodes['atomic_numbers']
        
        num_atoms = len(f)
        f_padded = np.pad(f, ((0, Nmax - num_atoms), (0, 0)), mode='constant', constant_values=0)
        p_padded = np.pad(p, ((0, Nmax - num_atoms), (0, 0)), mode='constant', constant_values=0)
        z_padded = np.pad(z, ((0, Nmax - num_atoms)), mode='constant', constant_values=0)
        nm = np.where(z_padded > 0, True, False)

        all_forces.append(f_padded)
        all_positions.append(p_padded)
        all_numbers.append(z_padded)
        all_energies.append(e)
        all_nodes_masks.append(nm)

    # Save to .npz file
    np.savez(
        save_dir / f'{filename}.npz',
        positions=np.stack(all_positions),
        atomic_numbers=np.stack(all_numbers),
        energy=np.stack(all_energies),
        forces=np.stack(all_forces),
        node_mask=np.stack(all_nodes_masks)
    )


if __name__ == '__main__':
    app.run(main)

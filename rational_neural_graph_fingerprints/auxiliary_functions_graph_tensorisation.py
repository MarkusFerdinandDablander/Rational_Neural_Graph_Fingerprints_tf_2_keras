''' 
PART_2

Auxiliary functions to generate graph tensor features for molecules given in smiles form.

Source:

The code in this part (i.e. PART 2 of this script) is adapted from

    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/preprocessing.py

Code adaptation to produce the version below was done by Markus Ferdinand Dablander, DPhil (= PhD) student at Mathematical Institute, Oxford University, August 2020.
'''


# import packages
import numpy as np
from rdkit import Chem
from . import auxiliary_functions_atom_bond_features as afabf



def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    ''' 
    Padds one axis of an array to a new size

    This is just a wrapper for np.pad, more usefull when only padding a single axis

    # Arguments:
        array: the array to pad
        new_size: the new size of the specified axis
        axis: axis along which to pad
        pad_value: pad value,
        pad_right: boolean, pad on the right or left side

    # Returns:
        padded_array: np.array

    '''
    
    
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(axis, array.shape[axis], new_size)
    pad_width = [(0,0)]*len(array.shape)

    # pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width = pad_width, mode = 'constant', constant_values = pad_value)


def tensorise_smiles(smiles, max_degree = None, max_atoms=None):
    '''
    Takes a list of smiles and turns the graphs in tensor representation.

    # Arguments:
        smiles: a list (or iterable) of smiles representations
        max_atoms: the maximum number of atoms per molecule (to which all molecules will be padded), use `None` for automatic inference of minimum
        max_degree: max_atoms: the maximum number of neigbours per atom that each molecule can have (to which all molecules will be padded), use `None`for automatic inference of minimum

    # Returns:
        atom_tensor: np.array, An atom feature np.array of shape (num_molecules, max_atoms, atom_features).
        bond_tensor: np.array, A bonds np.array of shape (num_molecules, max_atoms, max_neighbours).
        edge_tensor: np.array, A connectivity array of shape (num_molecules, max_atoms, max_neighbours, bond_features).
        atom__existence_tensor: np.arrary, a 1d array (for each molecule) containing a 1 in index i if the ith atom actually exists and a 0 otherwise. Of shape (num_molecules, max_atoms). The atoms are associated with indices as specified by atom_tensor. 
        
    The array atom_tensor can be seen as a sequence of arrays A_m of shape (max_atoms, atom_features), one array A_m for each molecule m.
    Let a be the number of atoms in m (i.e. the number of "1" values in the associated 1d slice of atom_existence_tensor). Then the first a rows of A_m are rows which do not only contain zeros (since they contain the features of the atoms)
    All rows of A_m with index >= a only contain zeros.
    '''

    
    # import sizes
    n = len(smiles)
    n_atom_features = afabf.num_atom_features_func()
    n_bond_features = afabf.num_bond_features_func()

    # preallocate atom tensor with 0's and bond tensor with -1 (because of 0 index)
    # if max_degree or max_atoms is set to None (auto), initialise dim as small as possible (1)
    atom_tensor = np.zeros((n, max_atoms or 1, n_atom_features))
    bond_tensor = np.zeros((n, max_atoms or 1, max_degree or 1, n_bond_features))
    edge_tensor = -np.ones((n, max_atoms or 1, max_degree or 1), dtype=int)
    atom_counts_tensor = np.zeros((n,))

    for mol_ix, s in enumerate(smiles):

        # load mol, atoms and bonds
        mol = Chem.MolFromSmiles(s)
        assert mol is not None, 'Could not parse smiles {}'.format(s)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        # save number of atoms
        atom_counts_tensor[mol_ix] = mol.GetNumAtoms()

        # if max_atoms is exceeded, resize if max_atoms=None (auto), else raise exception
        if len(atoms) > atom_tensor.shape[1]:
            assert max_atoms is None, 'too many atoms ({0}) in molecule: {1}'.format(len(atoms), s)
            atom_tensor = padaxis(atom_tensor, len(atoms), axis=1)
            bond_tensor = padaxis(bond_tensor, len(atoms), axis=1)
            edge_tensor = padaxis(edge_tensor, len(atoms), axis=1, pad_value=-1)

        rdkit_ix_lookup = {}
        connectivity_mat = {}

        for atom_ix, atom in enumerate(atoms):
            
            # write atom features
            atom_tensor[mol_ix, atom_ix, : n_atom_features] = afabf.atom_features(atom)

            # store entry in idx
            rdkit_ix_lookup[atom.GetIdx()] = atom_ix

        # preallocate array with neighbour lists (indexed by atom)
        connectivity_mat = [ [] for _ in atoms]

        for bond in bonds:
            
            # lookup atom ids
            a1_ix = rdkit_ix_lookup[bond.GetBeginAtom().GetIdx()]
            a2_ix = rdkit_ix_lookup[bond.GetEndAtom().GetIdx()]

            # lookup how many neighbours are encoded yet
            a1_neigh = len(connectivity_mat[a1_ix])
            a2_neigh = len(connectivity_mat[a2_ix])

            # if max_degree is exceeded, resize if max_degree=None (auto), else raise exception
            new_degree = max(a1_neigh, a2_neigh) + 1
            if new_degree > bond_tensor.shape[2]:
                assert max_degree is None, 'too many neighours ({0}) in molecule: {1}'.format(new_degree, s)
                bond_tensor = padaxis(bond_tensor, new_degree, axis=2)
                edge_tensor = padaxis(edge_tensor, new_degree, axis=2, pad_value=-1)

            # store bond features
            bond_feats = np.array(afabf.bond_features(bond), dtype=int)
            bond_tensor[mol_ix, a1_ix, a1_neigh, :] = bond_feats
            bond_tensor[mol_ix, a2_ix, a2_neigh, :] = bond_feats

            # add to connectivity matrix
            connectivity_mat[a1_ix].append(a2_ix)
            connectivity_mat[a2_ix].append(a1_ix)

        # store connectivity matrix
        for a1_ix, neighbours in enumerate(connectivity_mat):
            degree = len(neighbours)
            edge_tensor[mol_ix, a1_ix, : degree] = neighbours

    # create atom_existence_tensor for collected atom_counts_tensor
    max_atoms = atom_tensor.shape[1]
    atom_existence_tensor = np.zeros(shape = (n, max_atoms), dtype = np.float32)
    
    for k in range(n):
        a = int(atom_counts_tensor[k])
        atom_existence_tensor[k, 0:a] = 1
        
    # set arrays to right data type for tensorflow just to be safe
    atom_tensor = np.array(atom_tensor, dtype = np.float32)
    bond_tensor = np.array(bond_tensor, dtype = np.float32)
    edge_tensor = np.array(edge_tensor, dtype = np.float32)
    atom_existence_tensor = np.array(atom_existence_tensor, dtype = np.float32)

    return (atom_tensor, bond_tensor, edge_tensor, atom_existence_tensor)
''' 
PART_3

Auxiliary functions for the implementation of neural convolutional graph layers in tf.keras.

Source:

The code in this part (i.e. PART 3 of this script) is adapted from

    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/utils.py
    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py
    
The code was upgraded to work with tensorflow 2.2.0, rdkit 2020.03.3.0, numpy 1.19.1 and pandas 1.1.1.
Code adaptation and upgrading was done by Markus Ferdinand Dablander, DPhil (= PhD) student at Mathematical Institute, Oxford University, August 2020.
'''


#import packages
import tensorflow as tf


def mol_shapes_to_dims(mol_tensors = None, mol_shapes = None):
    
    ''' 
    Helper function, returns dim sizes for molecule tensors given tensors or tensor shapes.
    
    Input: mol_tensors = (atom_tensor, bond_tensor, edge_tensor) or mol_shapes = (atom_tensor.shape, bond_tensor.shape, edge_tensor.shape)
    
    Output: (max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules)
    
    '''

    if mol_shapes == None:
        mol_shapes = [t.shape for t in mol_tensors]

    num_molecules0, max_atoms0, num_atom_features = mol_shapes[0]
    num_molecules1, max_atoms1, max_degree1, num_bond_features = mol_shapes[1]
    num_molecules2, max_atoms2, max_degree2 = mol_shapes[2]

    num_molecules_vals = [num_molecules0, num_molecules1, num_molecules2]
    max_atoms_vals = [max_atoms0, max_atoms1, max_atoms2]
    max_degree_vals = [max_degree1, max_degree2]

    assert len(set(num_molecules_vals))==1, 'num_molecules does not match within tensors (found: {})'.format(num_molecules_vals)
    assert len(set(max_atoms_vals))==1, 'max_atoms does not match within tensors (found: {})'.format(max_atoms_vals)
    assert len(set(max_degree_vals))==1, 'max_degree does not match within tensors (found: {})'.format(max_degree_vals)

    return (max_atoms1, max_degree1, num_atom_features, num_bond_features, num_molecules1)


def neighbour_lookup(atoms, edges, maskvalue = 0, include_self = True):
    ''' 
    Looks up the features of all neighbours of an atom, for a batch of molecules.
    
    Arguments:
    
        - atoms (tensor): of shape (num_molecules, max_atoms, num_atom_features)
        - edges (tensor): of shape (num_molecules, max_atoms, max_degree), contains neighbour indices and -1 as padding value
        - maskvalue (numerical): the maskingvalue that should be used for empty atoms or atoms that have no neighbours (does not affect the input maskvalue which should always be -1!)
        - include_self (bool): if True, the featurevector of each atom will be added to the list feature vectors of its neighbours
    
    Returns:
    
        - neigbour_features (tensor): of shape (num_molecules, max_atoms, max_degree + (1), num_atom_features) depending on the value of include_self
    '''

    
    # the lookup masking trick: We add 1 to all indices, converting the masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    
    # we then add a padding vector at index 0 by padding to the left of the lookup matrix with the value that the new mask should get
    paddings = tf.constant([[0, 0], [1, 0], [0, 0]])
    masked_atoms = tf.pad(atoms, paddings, "CONSTANT")

    # import dimensions
    atoms_shape = tf.shape(masked_atoms)
    num_molecules = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]
    edges_shape = tf.shape(masked_edges)
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (num_molecules, 1, 1)
    offset = tf.reshape(tf.cast(tf.range(num_molecules, dtype = 'int32'), dtype = masked_edges.dtype), offset_shape)
    offset *= tf.cast(lookup_size, dtype = offset.dtype)

    # apply offset to account for the fact that after reshape, all individual num_molecules indices will be combined into a single big index
    flattened_atoms = tf.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = tf.reshape(masked_edges + offset, (num_molecules, -1))

    # gather flattened
    flattened_result = tf.gather(flattened_atoms, tf.cast(flattened_edges, dtype='int32'))

    # unflatten result
    neigbour_features_shape = (num_molecules, max_atoms, max_degree, num_atom_features)
    neigbour_features = tf.reshape(flattened_result, neigbour_features_shape)

    if include_self == True:
        return tf.concat([tf.expand_dims(atoms, axis=2), neigbour_features], axis=2)
    
    return neigbour_features
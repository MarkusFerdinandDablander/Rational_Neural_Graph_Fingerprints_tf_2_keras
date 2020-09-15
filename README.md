# Rational_Neural_Graph_Fingerprints_tf_keras

This package contains an implementation of two new tf.Keras layers (in tf.keras from tensorflow >= 2.0) which correspond to the operators necessary for computing neural graph fingerprints with trainable rational activation functions. The code was written to work with tf.keras on the basis of tensorflow 2.2.0, rdkit 2020.03.3.0, numpy 1.19.1 and pandas 1.1.1.

The script tf_keras_layers_rational_neural_graph_convolutions offers the following three tf.keras layer classes (child classes of tf.keras.layers.layer):

-RationalLayer: A layer of trainable rational activation functions (implemented by and taken from Nicolas Boulle, see https://github.com/NBoulle/RationalNets and https://arxiv.org/abs/2004.01902,).

- DeepRationalNeuralFingerprintHidden: Takes the place of the operation of the hidden graph convolution in Duvenauds algorithm (see matrices H in paper), but is now a deep neural network 5 layers with trainable rational activation functions.

- RationalNeuralFingerprintOutput: Takes the place of the operation of the readout convolution in Duvenauds algorithm (see matrices W in paper), but is now a shallow neural network with 1 hidden layer with a trainable activation function.

The scripts auxiliary_functions_atom_bond_features, auxiliary_functions_graph_tensorion and auxiliary_functions_neural_graph_convolutions contain auxiliary functions which were largely taken from the keiser-lab implementation of the Duvenaud algorithm ( - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py). Only minor changes were made and thus this implementation operates within the graph tensorisation framework which was offered in the keiser-lab implementation. We thus copied parts of the readme of the keiser-lab implementation which still apply to this new implementation below.

















References and Final Remarks

The two tf.keras neural graph fingerprint layers were implemented by Markus Ferdinand Dablander, DPhil (= PhD) student at Mathematical Institute, Oxford University, August 2020. The tf.keras layer RationalLater was implemented by and taken from the Github profile from Nicolas Boulle https://github.com/NBoulle/RationalNets.

The implementation of both neural fingerprint layers is inspired by (but different from) the two tf.keras graph convolutional layer implementations which can be found in the keiser-lab implementation:

    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py

An addition input tensor "atoms_existence" was added to the framework to account for a subtle theoretical gap in previous implementations: 
atoms associated with a zero feature vector (which can theoretically happen after at least one convolution) AND with degree 0 can still exist and can thus not be ignored. 
As an example imagine a single carbon atom as input molecule whose atom feature vector gets mapped to zero in the first convolution. The previous implementations would from
this moment on treat the carbon atom as nonexistent and thus the molecule as empty.
'''

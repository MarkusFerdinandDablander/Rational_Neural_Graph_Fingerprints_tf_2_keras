''' 
PART_4

This part (i.e. PART_4 of this script) contains an implementation of two new tf.Keras layers (in tf.keras from tensorflow >= 2.0) which correspond to the operators necessary for computing neural fingerprints for molecular graphs.

The script offers the following three tf.keras layer classes (child classes of tf.keras.layers.layer):

- RationalLayer: A layer of trainable rational activation functions (see https://arxiv.org/abs/2004.01902, implemented by Nicolas Boulle https://github.com/NBoulle/RationalNets).

- DeepRationalNeuralFingerprintHidden: Takes the place of the operation of the hidden graph convolution in Duvenauds algorithm (see matrices H in paper), but is now a deep neural network 5 layers with trainable rational activation functions.

- RationalNeuralFingerprintOutput: Takes the place of the operation of the readout convolution in Duvenauds algorithm (see matrices W in paper), but is now a shallow neural network with 1 hidden layer with a trainable activation function.

The code was written to work with tf.keras on the basis of tensorflow 2.2.0, rdkit 2020.03.3.0, numpy 1.19.1 and pandas 1.1.1.

The two tf.keras neural fingerprint layers were implemented by Markus Ferdinand Dablander, DPhil (= PhD) student at Mathematical Institute, Oxford University, August 2020. The tf.keras layer RationalLater was implemented by and taken from the Github profile from Nicolas Boulle https://github.com/NBoulle/RationalNets.

The implementation of both neural fingerprint layers is inspired by (but different from) the two tf.keras graph convolutional layer implementations which can be found in the keiser-lab implementation:

	- https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py

An addition input tensor "atoms_existence" was added to the framework to account for a subtle theoretical gap in previous implementations: 
atoms associated with a zero feature vector (which can theoretically happen after at least one convolution) AND with degree 0 can still exist and can thus not be ignored. 
As an example imagine a single carbon atom as input molecule whose atom feature vector gets mapped to zero in the first convolution. The previous implementations would from
this moment on treat the carbon atom as nonexistent and thus the molecule as empty.
'''


# import packages
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, InputSpec, Dropout
from tensorflow.keras import initializers, regularizers, constraints
from . import auxiliary_functions_graph_tensorisation as afgt
from . import auxiliary_functions_neural_graph_convolutions as afngf





class RationalLayer(tf.keras.layers.Layer):
	"""Rational Activation function (originally implemented by Nicolas Boulle at https://github.com/NBoulle/RationalNets and slightly adapted by Markus Dablander).
	Rational Function as Activation Function:
	`f(x) = P(x) / Q(x),
	where the coefficients of P and Q are learned array with the same shape as x.
	# Input shape
		Arbitrary. Use the keyword argument `input_shape`
		(tuple of integers, does not include the samples axis)
		when using this layer as the first layer in a model.
	# Output shape
		Same shape as the input.
	# Arguments
		alpha_initializer: initializer function for the weights of the numerator P.
		beta_initializer: initializer function for the weights of the denominator Q.
		alpha_regularizer: regularizer for the weights of the numerator P.
		beta_regularizer: regularizer for the weights of the denominator Q.
		alpha_constraint: constraint for the weights of the numerator P.
		beta_constraint: constraint for the weights of the denominator Q.
		shared_axes: the axes along which to share learnable
			parameters for the activation function.
			For example, if the incoming feature maps
			are from a 2D convolution
			with output shape `(batch, height, width, channels)`,
			and you wish to share parameters across space
			so that each filter only has one set of parameters,
			set `shared_axes=[1, 2]`.
	Reference: Rational neural networks](https://arxiv.org/abs/2004.01902)"""

	def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0], 
				 alpha_regularizer = None, beta_regularizer = None, alpha_constraint = None, beta_constraint = None,
				 shared_axes = None, **kwargs):
		super(RationalLayer, self).__init__(**kwargs)
		self.supports_masking = True

		# Degree of rationals
		self.degreeP = len(alpha_initializer) - 1
		self.degreeQ = len(beta_initializer) - 1
		
		# Initializers for P
		self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
		self.alpha_regularizer = regularizers.get(alpha_regularizer)
		self.alpha_constraint = constraints.get(alpha_constraint)
		
		# Initializers for Q
		self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.beta_constraint = constraints.get(beta_constraint)
		
		if shared_axes is None:
			self.shared_axes = None
		elif not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)

	def build(self, input_shape):
		param_shape = list(input_shape[1:])
		if self.shared_axes is not None:
			for i in self.shared_axes:
				param_shape[i - 1] = 1
		
		self.coeffsP = []
		for i in range(self.degreeP+1):
			# Add weight
			alpha_i = self.add_weight(shape=param_shape,
									 name='alpha_%s'%i,
									 initializer=self.alpha_initializer[i],
									 regularizer=self.alpha_regularizer,
									 constraint=self.alpha_constraint)
			self.coeffsP.append(alpha_i)
			
		# Create coefficients of Q
		self.coeffsQ = []
		for i in range(self.degreeQ+1):
			# Add weight
			beta_i = self.add_weight(shape=param_shape,
									 name='beta_%s'%i,
									 initializer=self.beta_initializer[i],
									 regularizer=self.beta_regularizer,
									 constraint=self.beta_constraint)
			self.coeffsQ.append(beta_i)
		
		# Set input spec
		axes = {}
		if self.shared_axes:
			for i in range(1, len(input_shape)):
				if i not in self.shared_axes:
					axes[i] = input_shape[i]
					self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
					self.built = True

	def call(self, inputs, mask=None):
		# Evaluation of P
		outP = tf.math.polyval(self.coeffsP, inputs)
		# Evaluation of Q
		outQ = tf.math.polyval(self.coeffsQ, inputs)
		# Compute P/Q
		out = tf.math.divide(outP, outQ)
		return out

	def get_config(self):
		config = {
			'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
			'beta_regularizer': regularizers.serialize(self.beta_regularizer),
			'alpha_constraint': constraints.serialize(self.alpha_constraint),
			'beta_constraint': constraints.serialize(self.beta_constraint),
			'shared_axes': self.shared_axes
		}
		base_config = super(RationalLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		return input_shape








class DeepRationalNeuralFingerprintHidden(tf.keras.layers.Layer):
	"""
	Hidden layer in a neural graph convolution. This layer takes a graph as an input. The graph is represented as by
	four tensors. Similar to Duvenaud et. al., 2015, but with 5 layers, per default with trainable rational activation functions.
	If use_rational_activation = True, then trainable rational activation functions are used, otherwise "activation" is used.
	
	- The atoms tensor represents the features of the nodes.
	- The bonds tensor represents the features of the edges.
	- The edges tensor represents the connectivity (which atoms are connected to which)
	- the atoms_existence tensor represents how many atoms each molecule has (its atom count) in form of a binary 1d array.
	
	It returns a tensor containing the updated atom feature vectors for each molecule.
	
	Input: (atoms, bonds, edges, atoms_existence)
	
	- atoms: shape = (num_molecules, max_atoms, num_atom_features))
	- bonds: shape = (num_molecules, max_atoms, max_degree, num_bond_features))
	- edges: shape = (num_molecules, max_atoms, max_degree)
	- atoms_existence: shape = (num_molecules, max_atoms)
	
	Output: atoms_updated
	
	- atoms_updated: updated (i.e. convolved) atom features with shape = (num_molecules, max_atoms, conv_width))
	"""
	

	def __init__(self, 
				 conv_width,
				 activation = tf.keras.activations.relu,
				 use_rational_activation = True,
				 rational_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
				 rational_beta_initializer = [2.383, 0.0, 1.0],
				 rational_shared_axes = [1,2],
				 use_bias = True, 
				 kernel_initializer = tf.keras.initializers.GlorotUniform,
				 bias_initializer = tf.keras.initializers.Zeros,
				 dropout_rate_input = 0,
				 dropout_rate_hidden = 0,
				 **kwargs):
		
		super(DeepRationalNeuralFingerprintHidden, self).__init__(**kwargs)
		
		self.conv_width = conv_width
		self.activation = activation
		self.use_rational_activation = use_rational_activation
		self.rational_alpha_initializer = rational_alpha_initializer
		self.rational_beta_initializer = rational_beta_initializer
		self.rational_shared_axes = rational_shared_axes
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.dropout_rate_input = dropout_rate_input
		self.dropout_rate_hidden = dropout_rate_hidden

		self.number_of_layers = 5
		self.max_degree = None # value of max_degree is not known just yet, but note that it further down defines the number of weight matrices W_d


	def build(self, inputs_shape):

		# import dimensions
		(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules) = afngf.mol_shapes_to_dims(mol_shapes = inputs_shape[0:3])
		num_atom_bond_features = num_atom_features + num_bond_features
		
		# set value for attribute self.max_degree
		self.max_degree = max_degree

		# initialize dropout layers
		self.Drop_input = Dropout(self.dropout_rate_input)
		self.Drop_hidden_1 = Dropout(self.dropout_rate_hidden)
		self.Drop_hidden_2 = Dropout(self.dropout_rate_hidden)
		self.Drop_hidden_3 = Dropout(self.dropout_rate_hidden)
		self.Drop_hidden_4 = Dropout(self.dropout_rate_hidden)

		# generate trainable dense layers D_1 ..., D_5 and perhaps also trainable rational layers R_1,...,R_5

		if self.use_rational_activation == True:
			
			# initialize dense layers
			self.D_1 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_2 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_3 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_4 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_output = Dense(units = self.conv_width, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)

			# initialize rational layers
			self.R_1 = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)
			self.R_2 = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)
			self.R_3 = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)
			self.R_4 = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)
			self.R_output = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)
			
		else:

			# initialize dense layers
			self.D_1 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_2 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_3 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_4 = Dense(units = self.conv_width, use_bias = self.use_bias, activation = self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.D_output = Dense(units = self.conv_width, use_bias = self.use_bias, activation = self.activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)


	def call(self, inputs, mask=None):
		
		atoms = inputs[0] # atoms.shape = (num_molecules, max_atoms, num_atom_features )
		bonds = inputs[1] # bonds.shape = (num_molecules, max_atoms, max_degree, num_edge_features)
		edges = inputs[2] # edges.shape = (num_molecules, max_atoms, max_degree)
		atoms_existence = inputs[3] # atoms_existence.shape (num_molecules, max_atoms)
		atoms_existence = tf.reshape(atoms_existence, shape = (tf.shape(atoms_existence)[0], 1, tf.shape(atoms_existence)[1])) # reshape atoms_existence to dimension (num_molecules, 1, max_atoms)

		# import dimensions
		max_atoms = atoms.shape[1]
		num_atom_features = atoms.shape[-1]
		num_bond_features = bonds.shape[-1]
		
		# sum the edge features for each atom
		summed_bond_features = tf.math.reduce_sum(bonds, axis = -2) # summed_bond_features.shape = (num_molecules, max_atoms, num_edge_features)
		
		# for each atom, look up the features of it's neighbour
		neighbour_atom_features = afngf.neighbour_lookup(atoms, edges, include_self = True)

		# sum along degree axis to get summed neighbour features
		summed_atom_features = tf.reduce_sum(neighbour_atom_features, axis = -2) # summed_atom_features.shape = (num_molecules, max_atoms, num_atom_features)
		
		# concatenate the summed atom and bond features
		summed_atom_bond_features = tf.concat([summed_atom_features, summed_bond_features], axis = -1) # summed_atom_bond_features.shape = (num_molecules, max_atoms, num_atom_bond_features)

		# apply dense layers (and rational layers if necessary)
		if self.use_rational_activation == True:

			new_features = self.Drop_input(summed_atom_bond_features)
				
			new_features = self.D_1(new_features)
			new_features = self.R_1(new_features)
			new_features = self.Drop_hidden_1(new_features)

			new_features = self.D_2(new_features)
			new_features = self.R_2(new_features)
			new_features = self.Drop_hidden_2(new_features)
				
			new_features = self.D_3(new_features)
			new_features = self.R_3(new_features)
			new_features = self.Drop_hidden_3(new_features)
				
			new_features = self.D_4(new_features)
			new_features = self.R_4(new_features)
			new_features = self.Drop_hidden_4(new_features)
				
			new_features = self.D_output(new_features)
			new_features = self.R_output(new_features)

		else:

			new_features = self.Drop_input(summed_atom_bond_features)

			new_features = self.D_1(new_features)
			new_features = self.Drop_hidden_1(new_features)
				
			new_features = self.D_2(new_features)
			new_features = self.Drop_hidden_2(new_features)
				
			new_features = self.D_3(new_features)
			new_features = self.Drop_hidden_3(new_features)
				
			new_features = self.D_4(new_features)
			new_features = self.Drop_hidden_4(new_features)
				
			new_features = self.D_output(new_features)


		# new_features.shape = (num_molecules, max_atoms, self.conv_width)
		
		# finally set feature rows of atoms_updated which correspond to non-existing atoms to 0 via 0-1 binary multiplicative masking (this step is where we need atoms_existence)
		atoms_updated = tf.linalg.matrix_transpose(tf.linalg.matrix_transpose(new_features) * atoms_existence)

		return atoms_updated


	def get_config(self):
		
		base_config = super(NeuralFingerprintHidden, self).get_config()
		
		config = {'Number of Output Units': self.conv_width,
				'Activation Function if no Rational Function is Used': self.activation,
				'Rational Functions as Activation Functions': self.use_rational_activation,
				'Rational Alpha Initializer': self.rational_alpha_initializer,
				'Rational Beta Initializer': self.rational_beta_initializer,
				'Rational Function Axes for Weight Sharing': self.rational_shared_axes,
				'Usage of Bias Vector': self.use_bias,
				'Kernel Initalizer': self.kernel_initializer,
				'Bias Initializer': self.bias_initializer,
				'Number of Neural Layers': self.number_of_layers,
				'Input Layer Dropout Rate': self.dropout_rate_input,
				'Hidden Layer Dropout Rate': self.dropout_rate_hidden}

		return dict(list(config.items()) + list(base_config.items()))










class RationalNeuralFingerprintOutput(tf.keras.layers.Layer):
	"""
	Output layer in a neural graph convolution.
	Similar to Duvenaud et. al., 2015, but with an additional hidden layer, per default with trainable rational activation functions.
	If use_rational_hidden_activation = True, then trainable rational activation functions are used, otherwise "hidden_activation" is used.
	This layer takes a graph as an input. The graph is represented as by four tensors.
	
	- The atoms tensor represents the features of the nodes.
	- The bonds tensor represents the features of the edges.
	- The edges tensor represents the connectivity (which atoms are connected to which)
	- the atoms_existence tensor represents how many atoms each molecule has (its atom count) in form of a binary 1d array.
	
	It returns the layer-based neural graph fingeprint for the layer features specified by the input. The neural fingerprints of all layers need to be summed up
	to obtain the fingerprint of the whole molecule according to Duvenaud.
	
	Input: (atoms, bonds, edges, atoms_existence)
	
	- atoms: shape = (num_molecules, max_atoms, num_atom_features))
	- bonds: shape = (num_molecules, max_atoms, max_degree, num_bond_features))
	- edges: shape = (num_molecules, max_atoms, max_degree)
	- atoms_existence: shape = (num_molecules, max_atoms)
	
	Output: fp_layerwise
	
	- fp_layerwise: Neural fingerprint for the graph layer specified by input, with shape = (num_molecules, fp_length)
	"""
	
	
	def __init__(self,
				 hidden_length, 
				 fp_length,
				 hidden_activation = tf.keras.activations.relu,
				 final_activation = tf.keras.activations.softmax,
				 use_rational_hidden_activation = True,
				 rational_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
				 rational_beta_initializer = [2.383, 0.0, 1.0],
				 rational_shared_axes = [1,2],
				 use_bias = True, 
				 kernel_initializer = tf.keras.initializers.GlorotUniform,
				 bias_initializer = tf.keras.initializers.Zeros,
				 dropout_rate_input = 0,
				 dropout_rate_hidden = 0,
				 **kwargs):
		
		super(RationalNeuralFingerprintOutput, self).__init__(**kwargs)
		
		self.hidden_length = hidden_length
		self.fp_length = fp_length
		self.hidden_activation = hidden_activation
		self.final_activation = final_activation
		self.use_rational_hidden_activation = use_rational_hidden_activation
		self.rational_alpha_initializer = rational_alpha_initializer
		self.rational_beta_initializer = rational_beta_initializer
		self.rational_shared_axes = rational_shared_axes
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.dropout_rate_input = dropout_rate_input
		self.dropout_rate_hidden = dropout_rate_hidden

		
	def build(self, inputs_shape):

		# import dimensions
		(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules) = afngf.mol_shapes_to_dims(mol_shapes = inputs_shape[0:3])
		num_atom_bond_features = num_atom_features + num_bond_features

		# initialize trainable layers

		self.Drop_input = Dropout(self.dropout_rate_input)
		self.Drop_hidden = Dropout(self.dropout_rate_hidden)

		if self.use_rational_hidden_activation == True:

			self.D_1 = Dense(units = self.hidden_length, use_bias = self.use_bias, activation = None, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)
			self.R = RationalLayer(shared_axes = self.rational_shared_axes, alpha_initializer = self.rational_alpha_initializer, beta_initializer = self.rational_beta_initializer)

		else:

			self.D_1 = Dense(units = self.hidden_length, use_bias = self.use_bias, activation = self.hidden_activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)

		self.D_2 = Dense(units = self.fp_length, use_bias = self.use_bias, activation = self.final_activation, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)

	   
	def call(self, inputs, mask = None):
		
		atoms = inputs[0] # atoms.shape = (num_molecules, max_atoms, num_atom_features )
		bonds = inputs[1] # bonds.shape = (num_molecules, max_atoms, max_degree, num_edge_features)
		edges = inputs[2] # edges.shape = (num_molecules, max_atoms, max_degree)
		atoms_existence = inputs[3] # atoms_existence.shape (num_molecules, max_atoms)
		atoms_existence = tf.reshape(atoms_existence, shape = (tf.shape(atoms_existence)[0], 1, tf.shape(atoms_existence)[1])) # reshape atoms_existence to dimension (num_molecules, 1, max_atoms)

		# import dimensions
		max_atoms = atoms.shape[1]
		num_atom_features = atoms.shape[-1]
		num_bond_features = bonds.shape[-1]

		# sum the edge features for each atom
		summed_bond_features = tf.math.reduce_sum(bonds, axis=-2) # summed_bond_features.shape = (num_molecules, max_atoms, num_edge_features)

		# concatenate the atom features and summed bond features
		summed_atom_bond_features = tf.concat([atoms, summed_bond_features], axis=-1) # summed_atom_bond_features.shape = (num_molecules, max_atoms, num_atom_bond_features)

		if self.use_rational_hidden_activation == True:

			# apply first dropout layer
			neural_fp_atomwise = self.Drop_input(summed_atom_bond_features)

			# apply first trainable dense layer
			neural_fp_atomwise = self.D_1(neural_fp_atomwise)

			# apply second dropout layer
			neural_fp_atomwise = self.Drop_hidden(neural_fp_atomwise)

			# apply trainable rational layer
			neural_fp_atomwise = self.R(neural_fp_atomwise)

			# apply second trainable dense layer
			neural_fp_atomwise = self.D_2(neural_fp_atomwise)

		else:

			# apply first dropout layer
			neural_fp_atomwise = self.Drop_input(summed_atom_bond_features)

			# apply first trainable dense layer
			neural_fp_atomwise = self.D_1(neural_fp_atomwise)

			# apply second dropout layer
			neural_fp_atomwise = self.Drop_hidden(neural_fp_atomwise)

			# apply second trainable dense layer
			neural_fp_atomwise = self.D_2(neural_fp_atomwise)

		# set feature rows of neural_fp_atomwise which correspond to non-existing atoms to 0 via 0-1 binary multiplicative masking (this step is where we need atoms_existence)
		neural_fp_atomwise_masked = tf.linalg.matrix_transpose(tf.linalg.matrix_transpose(neural_fp_atomwise) * atoms_existence)
		
		# add up all atomwise neural fingerprints of all existing atoms in each molecule to obtain the layerwise neural fingerprint
		neural_fp_layerwise = tf.math.reduce_sum(neural_fp_atomwise_masked, axis=-2) # neural_fp_layerwise.shape = (num_molecules, self.fp_length)

		return neural_fp_layerwise
	
	
	def get_config(self):
		
		base_config = super(NeuralFingerprintOutput, self).get_config()
		
		config = {'Number of Hidden Units': self.hidden_length,
				'Number of Output Units (Neural Fingerprint Length)': self.fp_length,
				'Hidden Activation Function if no Rational Function is Used': self.hidden_activation,
				'Final Activation Function': self.final_activation,
				'Rational Functions as Hidden Activation Functions': use_rational_hidden_activation,
				'Rational Alpha Initializer': self.rational_alpha_initializer,
				'Rational Beta Initializer': self.rational_beta_initializer,
				'Rational Function Axes for Weight Sharing': self.rational_shared_axes,
				'Usage of Bias Vector': self.use_bias,
				'Kernel Initalizer': self.kernel_initializer,
				'Bias Initializer': self.bias_initializer,
				'Input Layer Dropout Rate': self.dropout_rate_input,
				'Hidden Layer Dropout Rate': self.dropout_rate_hidden}

		return dict(list(config.items()) + list(base_config.items()))

from .base_organizer import OrganizerModel
import numpy as np, theano, theano.tensor as T
from .utils import number_of_branches_per_depth

REAL = theano.config.floatX

class BiasOrganizerModel(OrganizerModel):
	"""
	Multiplicative interactions between document and classification
	matrix induce specialization of the model.

	Note: Could add dropout noise to make this more specific.
	
	"""

	def _create_theano_variables(self):
		super()._create_theano_variables()

		self.document_bias_matrix = theano.shared(
			np.zeros([self.document_size, number_of_branches_per_depth(self.tree_depth)]).astype(REAL),
			name = 'document_bias_matrix')

		self.projection_matrix = theano.shared(
			1./self.size * np.random.randn(self.size).astype(REAL),
			name = 'projection_matrix')
		
		self.params.append(self.projection_matrix)
		self.document_indexed_params.add(self.document_bias_matrix)


	def projection_function(self, indices, document_index, branch_index):
		proj_mat = self.document_matrix[document_index, branch_index] * self.projection_matrix
		word_mat = self.model_matrix[indices].mean(axis=0)
		proj = T.dot(
				word_mat,
				proj_mat
				)
		return T.nnet.sigmoid(proj + self.bias_vector + self.document_bias_matrix[document_index, branch_index])
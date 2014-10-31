from gradient_optimizers import GradientModel
import theano, theano.tensor as T, numpy as np
from collections import OrderedDict
REAL = theano.config.floatX

class OrganizerModel(GradientModel):
    """
    Language model built around the idea of binary search.
    Method is simple, take any input text, assign a tree
    to the text by recursively cutting it in halves. Take
    random words inside and train the vector to separate
    the words into the tree (hierarchical softmax in a
    minimalistic setting)
    
    """
    
    def projection_function(self, indices, document_index):
        """
        Project word using document into 2 classes
        using a bias vector.
        """
        
        proj = T.dot(
                self.model_matrix[indices],
                self.document_matrix[document_index])
        return T.nnet.softmax(proj + self.bias_vector)
        
    def cost_function(self, projection, label):
        """
        Collect error by comparing the decision of the network
        with actual position of word in document.
        """
        return T.nnet.categorical_crossentropy(projection, label)
        
    def _create_theano_variables(self):
        
        self.model_matrix = theano.shared(
            np.random.randn(self.vocabulary_size, self.size).astype(REAL),
            name = 'model_matrix')
        
        self.document_matrix = theano.shared(
            np.random.randn(self.document_size, self.size, 2).astype(REAL),
            name = 'model_matrix')
        
        self.bias_vector = theano.shared(np.zeros(2, dtype = REAL), name='bias_vector')
        
        self.params.append(self.bias_vector)
        self.params.append(self.model_matrix)
        self.indexed_params.add(self.model_matrix)
        self.document_indexed_params.add(self.document_matrix)
        
    
    def __init__(self,
                 store_max_updates = False,
                 l2_regularization = None,
                 theano_mode = 'FAST_RUN',
                 disconnected_inputs = 'ignore',
                 update_fun = True,
                 learning_rate = 0.035,
                 size = 50,
                 tree_depth = 3,
                 vocabulary_size = 5000,
                 document_size = 500,
                 update_function = 'adagradclipped'):
        
        self.disconnected_inputs = disconnected_inputs
        self.theano_mode         = theano_mode
        self.store_max_updates   = store_max_updates
        self.size                = size
        self.vocabulary_size     = vocabulary_size
        self.document_size       = (2 ** (tree_depth)) * document_size
        self.tree_depth          = tree_depth
        self.learning_rate       = theano.shared(np.float32(learning_rate), name='learning_rate')
        setattr(self, 'params', [])
        setattr(self, 'indexed_params', set())
        setattr(self, 'document_indexed_params', set())
        
        self._l2_regularization  = True if l2_regularization is not None else False
        
        if l2_regularization is not None:
            self._l2_regularization_parameter = theano.shared(np.float64(l2_regularization).astype(REAL), name='l2_regularization_parameter')
        
        self._create_theano_variables()
        
        self._select_update_mechanism(update_function)
        if update_fun:
            self.create_update_fun()
            
    def _create_clipped_adagrad_update_mechanism(self):

        if self.store_max_updates:
            self.max_update_size = theano.shared(np.zeros(len(self.params), dtype=REAL), 'max_update_size')
        self._additional_params = {}

        for param in self.params:
            self._additional_params[param] = theano.shared(np.ones_like(param.get_value(borrow=True)), name="%s_statistic" % (param.name))

        self.clip_range      = theano.shared(np.float32(10))
        indices              = T.ivector('indices')
        document_index       = T.iscalar('document_index')
        label                = T.ivector('labels')
        
        class_projection     = self.projection_function(indices, document_index)
        
        cost                 = self.cost_function(class_projection, label).sum()

        if self._l2_regularization:
            cost += self.l2_regularization(indices)
        
        gparams              = T.grad(cost, self.params, disconnected_inputs = self.disconnected_inputs )
        updates              = OrderedDict()
        reset_updates        = OrderedDict()
        
        i = 0
        if self.store_max_updates:
            updates[self.max_update_size] = self.max_update_size

        for param, gparam in zip(self.params, gparams):

            if self._skip_update_param(param):
                continue

            if param in self.indexed_params:
                # the statistic gets increased by the squared gradient:
                updates[self._additional_params[param]] = T.inc_subtensor(self._additional_params[param][indices], gparam[indices] ** 2)
                reset_updates[self._additional_params[param]] = T.ones_like(self._additional_params[param])

                if self.store_max_updates:
                    updates[self.max_update_size] = T.set_subtensor(updates[self.max_update_size][i], T.maximum(self.max_update_size[i], gparam[indices].max()))
                gparam = T.clip(gparam[indices], -self.clip_range, self.clip_range)
                # this normalizes the learning rate:
                updates[param] = T.inc_subtensor(param[indices], - (self.learning_rate / T.sqrt(updates[self._additional_params[param]][indices])) * gparam)
            elif param in self.document_indexed_params:
                # the statistic gets increased by the squared gradient:
                updates[self._additional_params[param]] = T.inc_subtensor(self._additional_params[param][indices], gparam[document_index] ** 2)
                reset_updates[self._additional_params[param]] = T.ones_like(self._additional_params[param])

                if self.store_max_updates:
                    updates[self.max_update_size] = T.set_subtensor(updates[self.max_update_size][i], T.maximum(self.max_update_size[i], gparam[document_index].max()))
                gparam = T.clip(gparam[document_index], -self.clip_range, self.clip_range)
                # this normalizes the learning rate:
                updates[param] = T.inc_subtensor(param[document_index], - (self.learning_rate / T.sqrt(updates[self._additional_params[param]][document_index])) * gparam)
            else:
                # the statistic gets increased by the squared gradient:
                updates[self._additional_params[param]] = self._additional_params[param] + (gparam ** 2)
                reset_updates[self._additional_params[param]] = T.ones_like(self._additional_params[param])

                if self.store_max_updates:
                    updates[self.max_update_size] = T.set_subtensor(updates[self.max_update_size][i], T.maximum(self.max_update_size[i], gparam.max()))
                gparam = T.clip(gparam, -self.clip_range, self.clip_range)
                # this normalizes the learning rate:
                updates[param] = param - (self.learning_rate / T.sqrt(updates[self._additional_params[param]])) * gparam

            i+=1

        self.update_fun      = theano.function([indices, document_index, label], cost, updates = updates, mode = self.theano_mode)
        self.reset_adagrad   = theano.function([], updates = reset_updates, mode = self.theano_mode)
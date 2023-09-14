import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    _gradient_weights = None

    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size, self.output_size)  ## X.T* W.T = Y.T
        self.bias = np.random.rand(1, self.output_size)
        self._weights = np.concatenate([self.weights, self.bias], axis=0)
        self.input_tensor = None
        self._optimizer = None
        #self.regularization = None
        self.regularization_loss = 0

    def forward(self, input_tensor):
        self.input_size = input_tensor.shape[1]
        columns = input_tensor.shape[0]
        one = np.ones((columns, 1))
        self.input_tensor = np.concatenate([input_tensor, one], axis=1)
        output = np.dot(self.input_tensor, self._weights)
        self.output_size = output.shape[1]
        return output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        if self._optimizer.regularizer is not None:
            self.regularization = self.optimizer.regularizer


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def weights(self):
        # self.weights_optimizer = copy.deepcopy(self.optimizer)
        return self._weights

    @weights.setter
    def weights(self, weights):
        # self.weights_optimizer = copy.deepcopy(self.optimizer)
        self._weights = weights

    def backward(self, error_tensor):
        error = np.dot(error_tensor, self._weights.T)
        error_previous = error[:, :self.input_size]
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self._weights, self._gradient_weights)

        return error_previous

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.input_size, self.output_size)
        bias_shape = (1, self.output_size)
        fan_in = self.input_size
        fan_out = self.output_size
        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)
        self.weights = np.concatenate([self.weights, self.bias], axis=0)

    def calculate_regularization_loss(self):
        if self.regularization is not None:
            #print("calculate regu. in fully connected")
            #self.regularization = self.optimizer.regularizer
            regularization_loss = self.regularization.norm(self.weights)
        else:
            regularization_loss = 0

        return regularization_loss
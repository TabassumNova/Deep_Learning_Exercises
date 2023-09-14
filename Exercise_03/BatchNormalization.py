from .Base import BaseLayer
import numpy as np
from .Helpers import compute_bn_gradients
class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        BaseLayer.__init__(self)
        self.trainable = True
        self.channels = channels
        self.bias = np.zeros(self.channels) # same length as channels
        self.weights = np.ones(self.channels)  # beta, gama, mu, sigma
        self.eps = np.finfo(float).eps
        self.mean = np.array([])     # for every batch-> every channel-> compute mu, sigma
        self.variance = np.array([])
        self.alpha = 0.8
        self.mu_tilde = None
        self.sigma_tilde = 0
        self.input_tensor = None
        self.X_tilde = None
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.tensor_status = None
        self.image_tensor_shape = None
        self.reshape1 = None
        self.reshape2 = None
        self.reshape3 = None
        self.error_status = None
        self.past_mu_tilde = 0
        self.past_sigma_tilde = 0


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            self.image_tensor_shape = tensor.shape
            B = tensor.shape[0]
            H = tensor.shape[1]
            M = tensor.shape[2]
            N = tensor.shape[3]
            self.reshape1 = tensor.reshape(B, H, -1)
            self.reshape2 = np.transpose(self.reshape1, (0, 2, 1))
            self.reshape3 = self.reshape2.reshape(-1, H)
            reshaped_tensor = self.reshape3
        elif len(tensor.shape) == 2:
            reshape1 = tensor.reshape(self.reshape2.shape)
            reshape2 = np.transpose(reshape1, (0, 2, 1))
            reshape3 = reshape2.reshape(self.image_tensor_shape)
            reshaped_tensor = reshape3

        return reshaped_tensor

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        tensor = input_tensor
        if len(self.input_tensor.shape) == 4:
            self.tensor_status = 'Image'
            tensor = self.reformat(input_tensor)
            self.input_tensor = tensor
        else:
            pass

        if self.testing_phase == True:
            self.mean = self.mu_tilde
            self.variance = self.sigma_tilde
        else:
            self.mean = np.mean(tensor, axis=0).reshape(1,-1)     # mu_b
            self.variance = np.var(tensor, axis=0).reshape(1,-1)  # sigma_b
            # Moving Average Estimation
            if self.mu_tilde is None:
                self.mu_tilde = self.mean
                self.sigma_tilde = self.variance
            else:
                self.mu_tilde = self.alpha * self.mu_tilde + (1 - self.alpha) * self.mean
                self.sigma_tilde = self.alpha * self.sigma_tilde + (1 - self.alpha) * self.variance

        v = np.sqrt(np.add(self.variance, self.eps))
        self.X_tilde = (tensor - self.mean) / v
        output = self.weights * self.X_tilde + self.bias

        if self.tensor_status == 'Image':
            output = self.reformat(output)

        return output


    def backward(self, error_tensor):

        error = error_tensor
        if len(error_tensor.shape) == 4:
            self.error_status = 'Image'
            error = self.reformat(error_tensor)

        # Gradient w.r.to Input
        error_prev = compute_bn_gradients(error, self.input_tensor,
                                          self.weights, self.mean, self.variance)
        output = error_prev

        if self.error_status == 'Image':
            output = self.reformat(error_prev)

        # Gradient w.r.to Weights
        self.gradient_weights = np.sum(error * self.X_tilde, axis=0)
        self.gradient_bias = np.sum(error, axis=0)

        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)

        return output


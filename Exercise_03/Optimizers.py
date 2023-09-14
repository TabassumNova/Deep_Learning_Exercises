from typing import Any, Union
import copy
import numpy as np
import sys

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):

    def __init__(self, learning_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight = weight_tensor
        if self.regularizer:
            subgrad_weights = self.regularizer.calculate_gradient(weight)
            weight = weight_tensor - self.learning_rate * subgrad_weights
        updated_weight_tensor = weight - self.learning_rate * gradient_tensor

        return updated_weight_tensor

class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.past_gradients = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight = weight_tensor
        if self.regularizer:
            subgrad_weights = self.regularizer.calculate_gradient(weight)
            weight = weight_tensor - self.learning_rate * subgrad_weights
        vk = np.multiply(self.momentum_rate, self.past_gradients) + np.multiply(self.learning_rate, gradient_tensor)
        updated_weight_tensor = weight - vk
        self.past_gradients = vk
        return updated_weight_tensor

class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.past_vk = 0
        self.past_rk = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight = weight_tensor
        if self.regularizer:
            subgrad_weights = self.regularizer.calculate_gradient(weight)
            weight = weight_tensor - self.learning_rate * subgrad_weights
        gk = gradient_tensor
        vk = np.multiply(self.mu, self.past_vk) + np.multiply((1 - self.mu), gk)
        rk = np.multiply(self.rho, self.past_rk) + np.multiply((1-self.rho)*gk, gk)
        vk_hat = vk / (1 - (self.mu ** self.k))
        rk_hat = rk / (1 - (self.rho ** self.k))
        updated_weight_tensor = weight - ((self.learning_rate * (vk_hat + sys.float_info.epsilon))/
                                                 (np.sqrt(rk_hat) + sys.float_info.epsilon))
        self.past_vk = vk
        self.past_rk = rk
        self.k = self.k + 1

        return updated_weight_tensor



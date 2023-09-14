from .Base import BaseLayer
import numpy as np
import math
class Sigmoid(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        self.fx = None
        self.fx_inverse = None

    def forward(self, input_tensor):
        e = np.exp(-input_tensor)
        self.fx = 1 / (1 + e)
        return self.fx

    def backward(self, error_tensor):
        self.fx_inverse = self.fx * (1 - self.fx)
        error = self.fx_inverse * error_tensor
        return error

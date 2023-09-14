from .Base import BaseLayer
import numpy as np
class TanH(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        self.fx = None
        self.fx_inverse = None

    def forward(self, input_tensor):
        self.fx = np.tanh(input_tensor)
        return self.fx

    def backward(self, error_tensor):
        self.fx_inverse = 1 - self.fx ** 2
        error = self.fx_inverse * error_tensor
        return error
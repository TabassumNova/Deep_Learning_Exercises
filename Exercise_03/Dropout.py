from .Base import BaseLayer
import numpy as np
class Dropout(BaseLayer):
    def __init__(self, probability):
        BaseLayer.__init__(self)
        self.probability = probability
        self.p = None

    def forward(self, input_tensor):

        if self.testing_phase == False:
            self.p = np.random.rand(input_tensor.shape[-2], input_tensor.shape[-1]) >= (1 - self.probability)
            dropout = input_tensor * self.p
            output = dropout * (1 / self.probability)
        else:
            output = input_tensor

        return output

    def backward(self, error_tensor):
        out = error_tensor * (1 / self.probability)
        output = out * self.p
        return output

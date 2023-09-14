import numpy as np
from .Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)

    def forward(self, input_tensor):
        self.inpur_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        output = []
        for img in range(0, batch_size):
            output.append(input_tensor[img].flatten())
        output_tensor = np.array(output)
        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        error_prev = []
        for img in range(0, batch_size):
            reshaped_img = error_tensor[img].reshape((self.inpur_tensor[img].shape))
            error_prev.append(reshaped_img)
        error_prev_array = np.array(error_prev)
        return error_prev_array
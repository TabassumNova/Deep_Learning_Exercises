import numpy as np
class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant
    def initialize(self, weights_shape,fan_in, fan_out):
        initialized_object = np.full(weights_shape, self.constant)
        return initialized_object
class UniformRandom:
    def initialize(self, weights_shape,fan_in, fan_out):
        num = fan_in * fan_out
        initialized_object = np.random.uniform(0,1,num).reshape(weights_shape)
        return initialized_object
class Xavier:
    def initialize(self, weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(np.add(fan_in, fan_out))
        #num = fan_in * fan_out
        num = 1
        for x in weights_shape:
            num = (num * x)
        initialized_object = np.random.normal(0, sigma, num).reshape(weights_shape)
        return initialized_object
class He:
    def initialize(self, weights_shape,fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in)
        #num = fan_in * fan_out
        num = 1
        for x in weights_shape:
            num = (num * x)
        initialized_object = np.random.normal(0, sigma, num).reshape(weights_shape)
        return initialized_object
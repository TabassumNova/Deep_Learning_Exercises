import numpy as np
from .Base import BaseLayer
from .TanH import TanH
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected
import copy
class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros(self.hidden_size)
        self.output_size = output_size
        self._memorize = False
        self.fullyconnected_objects_1 = []
        self.fullyconnected_objects_2 = []
        self.tanh_objects = []
        self.sigmoid_objects = []
        self.weights1 = np.random.rand(self.input_size + self.hidden_size, self.hidden_size)
        self.bias = np.random.rand(1, self.hidden_size)
        self._weights = np.concatenate([self.weights1, self.bias], axis=0)
        self.weights_hy1 = np.random.rand(self.hidden_size, self.output_size)
        self.bias_y = np.random.rand(1, self.output_size)
        self._weights_hy = np.concatenate([self.weights_hy1, self.bias_y], axis=0)
        self._optimizer = None
        self.weights_optimizer = None
        self.weights_hy_optimizer = None
        self.gradient_weights_hy = None
        self.gradient_weights = None
        self.regularizer = None
        self.regularization_loss = None


    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_weights_hy(self):
        return self._gradient_weights_hy

    @gradient_weights_hy.setter
    def gradient_weights_hy(self, gradient_weights_hy):
        self._gradient_weights_hy = gradient_weights_hy

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def weights_hy(self):
        return self._weights_hy

    @weights_hy.setter
    def weights_hy(self, weights_hy):
        self._weights_hy = weights_hy

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.weights_hy_optimizer = copy.deepcopy(self._optimizer)

        if self._optimizer.regularizer is not None:
            self.regularization = self.optimizer.regularizer

    def forward(self, input_tensor):

        batch = input_tensor.shape[0]
        y_t = []
        i = 0
        self.fullyconnected_objects_1 = []
        self.fullyconnected_objects_2 = []
        self.tanh_objects = []
        self.sigmoid_objects = []

        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        else:
            pass

        for time in range(0, batch):

            Xt_tilde = np.concatenate([input_tensor[time].reshape(1,-1), self.hidden_state.reshape(1,-1)], axis=1)
            input_size1 = int(Xt_tilde.shape[-1])
            output_size1 = int(self.hidden_state.shape[-1])
            fullyconnected1 = FullyConnected(input_size1, output_size1)
            fullyconnected1.weights = np.copy(self._weights)
            full_forward1 = fullyconnected1.forward(Xt_tilde)
            self.fullyconnected_objects_1.append(fullyconnected1)
            tanh_obj = TanH()
            output1 = tanh_obj.forward(full_forward1)
            self.tanh_objects.append(tanh_obj)
            self.hidden_state = output1

            input_size2 = output_size1
            output_size2 = self.output_size
            fullyconnected2 = FullyConnected(input_size2, output_size2)
            fullyconnected2.weights = np.copy(self._weights_hy)
            full_forward2 = fullyconnected2.forward(self.hidden_state)
            self.fullyconnected_objects_2.append(fullyconnected2)
            sigmoid_obj = Sigmoid()
            output2 = sigmoid_obj.forward(full_forward2).squeeze()
            self.sigmoid_objects.append(sigmoid_obj)
            y_t.append(output2)
        Yt = np.array(y_t)

        return Yt

    def backward(self, error_tensor):
        batch = error_tensor.shape[0]
        grad_hidden = np.zeros(self.hidden_size)
        all_grad_Xt = []
        grad_Why = np.zeros(self._weights_hy.shape)
        grad_Wh = np.zeros(self._weights.shape)
        for time in reversed(range(batch)):
            sigmoid_backward = self.sigmoid_objects[time].backward(error_tensor[time])
            fullyconnected2_backward = self.fullyconnected_objects_2[time].backward(sigmoid_backward)
            grad_Why_temp = grad_Why + self.fullyconnected_objects_2[time].gradient_weights  # gradient w.r.to weights (Why)
            grad_Why = grad_Why_temp
            copy_backward = grad_hidden + fullyconnected2_backward
            tanh_backward = self.tanh_objects[time].backward(copy_backward)
            fullyconnected1_backward = self.fullyconnected_objects_1[time].backward(tanh_backward).squeeze()
            grad_Wh_temp = grad_Wh + self.fullyconnected_objects_1[time].gradient_weights  # gradient w.r.to weights (Wh)
            grad_Wh = grad_Wh_temp
            grad_Xt = fullyconnected1_backward[0: self.input_size]  # gradient w.r.to input
            all_grad_Xt.append(grad_Xt)
            grad_hidden = fullyconnected1_backward[self.input_size: self.input_size+self.hidden_size]

        all_grad_Xt_array = np.array(all_grad_Xt)
        error_prev = np.flip(all_grad_Xt_array, 0)

        self.gradient_weights_hy = grad_Why
        self.gradient_weights = grad_Wh
        if self._optimizer != None:
            self._weights = self.weights_optimizer.calculate_update(self._weights, self.gradient_weights)
            self._weights_hy = self.weights_hy_optimizer.calculate_update(self._weights_hy, self.gradient_weights_hy)

        return error_prev

    def initialize(self, weights_initializer, bias_initializer):

        weights_shape = self.weights1.shape
        bias_shape = self.bias.shape
        weights_hy_shape = self.weights_hy1.shape
        bias_y_shape = self.bias_y.shape
        w_fan_in = self.weights1.shape[0]
        w_fan_out = self.weights1.shape[1]
        b_fan_in = self.bias.shape[0]
        b_fan_out = self.bias.shape[1]
        why_fan_in = self.weights_hy1.shape[0]
        why_fan_out = self.weights_hy1.shape[1]
        by_fan_in = self.bias_y.shape[0]
        by_fan_out = self.bias_y.shape[1]
        self.weights1 = weights_initializer.initialize(weights_shape, w_fan_in, w_fan_out)
        self.bias = bias_initializer.initialize(bias_shape, b_fan_in, b_fan_out)
        self._weights = np.concatenate([self.weights1, self.bias], axis=0)
        self.weights_hy1 = weights_initializer.initialize(weights_hy_shape, why_fan_in, why_fan_out)
        self.bias_y = bias_initializer.initialize(bias_y_shape, by_fan_in, by_fan_out)
        self._weights_hy = np.concatenate([self.weights_hy1, self.bias_y], axis=0)

    def calculate_regularization_loss(self):

        if self.regularization is not None:
            regularization_loss = self.regularization.norm(self.weights)
        else:
            regularization_loss = 0

        return regularization_loss

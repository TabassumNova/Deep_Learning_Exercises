import copy
class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.loss_layer = None
        self.data_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None
        self._testing_phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        regularization_loss = 0
        for layers in self.layers:
            output = layers.forward(self.input_tensor)
            if layers.regularization is not None:
                regularization_loss_temp = regularization_loss + layers.calculate_regularization_loss()
                regularization_loss = regularization_loss_temp
            self.input_tensor = output
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor) + regularization_loss
        return loss

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for layers in reversed(self.layers):
            error = layers.backward(error)

    def append_layer(self, layer):

        self.layers.append(layer)
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)

    def train(self, iterations):
        for layers in self.layers:
            layers.testing_phase = False
        for i in range (0,iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layers in self.layers:
            layers.testing_phase = True
            output = layers.forward(input_tensor)
            input_tensor = output
        return output




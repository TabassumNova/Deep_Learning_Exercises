import copy
class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.loss_layer = None
        self.data_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layers in self.layers:
            output = layers.forward(self.input_tensor)
            self.input_tensor = output
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for layers in reversed(self.layers):
            error = layers.backward(error)

    def append_layer(self, layer):
        self.layers.append(layer)
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

    def train(self, iterations):
        for i in range (0,iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layers in self.layers:
            output = layers.forward(input_tensor)
            input_tensor = output
        return output




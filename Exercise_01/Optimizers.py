from typing import Any, Union
class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, wight_tensor, gradient_tensor):
        updated_wight_tensor = wight_tensor - self.learning_rate * gradient_tensor

        return updated_wight_tensor

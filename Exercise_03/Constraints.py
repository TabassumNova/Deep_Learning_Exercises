import numpy as np
class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        subgrad_weights = self.alpha * weights
        return subgrad_weights

    def norm(self, weights):
        w = np.linalg.norm(weights)
        regularization_loss = self.alpha * w * w
        return regularization_loss

class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        subgrad_weights = self.alpha * np.sign(weights)
        return subgrad_weights

    def norm(self, weights):
        w = np.sum(np.abs(weights))
        regularization_loss = self.alpha * np.abs(w)
        return regularization_loss
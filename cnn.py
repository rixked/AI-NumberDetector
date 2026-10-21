import numpy as np
class ConvolutionalNeuralNetwork:
    def __init__(self, iNeuron, hNeuron, oNeuron, learningRate):
            self.iNeuron = iNeuron
            self.hNeuron = hNeuron
            self.oNeuron = oNeuron
            self.learningRate = learningRate
            self.weight_ih = np.random.normal(0.0, pow(self.iNeuron, -0.5), (self.hNeuron, self.iNeuron))
            self.weight_ho = np.random.normal(0.0, pow(self.hNeuron, -0.5), (self.oNeuron, self.hNeuron))
    
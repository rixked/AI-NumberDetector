import numpy as np
class NeuronalNetwork:
    def __init__(self, iNeuron, hNeuron, oNeuron, learningRate):
        self.iNeuron = iNeuron
        self.hNeuron = hNeuron
        self.oNeuron = oNeuron
        self.learningRate = learningRate
        self.weight_ih = np.random.normal(0.0, pow(self.iNeuron, -0.5), (self.hNeuron, self.iNeuron))
        self.weight_ho = np.random.normal(0.0, pow(self.hNeuron, -0.5), (self.oNeuron, self.hNeuron))


    def activationFct(self, x):
        #TODO sigmoid
        return 1/(1+pow(np.e,-x))

    def train(self, train_image, train_label):
        fish = np.zeros(self.oNeuron) + 0.01
        fish[train_label]=0.99
        
        summen = np.dot(self.weight_ih, train_image)
        hidden = self.activationFct(summen)
        output = self.activationFct(np.dot(self.weight_ho, hidden))

        output_error = fish - output
        hidden_error = np.dot(self.weight_ho.T,output_error)

        a  = output_error * (output * (1 - output))
        change_ho = self.learningRate * np.outer(a,hidden)
        self.weight_ho = self.weight_ho + change_ho

        b = hidden_error * (hidden * (1 - hidden))
        change_ih = self.learningRate * np.outer(b,train_image)
        self.weight_ih = self.weight_ih + change_ih

    def getOutput(self, test_image):
        summen = np.dot(self.weight_ih, test_image)
        hidden = self.activationFct(summen)
        output = self.activationFct(np.dot(self.weight_ho, hidden))
        return output

    def checkImage(self, test_image):
        return np.argmax(self.getOutput(test_image))

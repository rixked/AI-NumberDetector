import numpy as np
import os
import nn


def convolve(image, filters):
    # slide every filter over the image -> one feature map per filter
    size = len(image) - len(filters[0]) + 1
    kernel_map = np.zeros((len(filters), size, size))
    for f in range(len(filters)):
        for r in range(size):
            for c in range(size):
                patch = image[r:r + len(filters[0]), c:c + len(filters[0])]
                kernel_map[f, r, c] = np.sum(filters[f] * patch)
    return kernel_map


def maxPool(kernel_map):
    # keep the largest value of every 2x2 block
    poolSize = len(kernel_map[0]) // 2
    pooled = np.zeros((len(kernel_map), poolSize, poolSize))
    for f in range(len(kernel_map)):
        for r in range(poolSize):
            for c in range(poolSize):
                block = kernel_map[f, r * 2: r * 2 + 2, c * 2: c * 2 + 2]
                pooled[f, r, c] = np.max(block)
    return pooled


def maxPoolBackward(kernel_map, pooled_error):
    # the error of each pooled value goes back only to where the maximum was
    map_error = np.zeros(kernel_map.shape)
    poolSize = len(kernel_map[0]) // 2
    for f in range(len(kernel_map)):
        for r in range(poolSize):
            for c in range(poolSize):
                block = kernel_map[f, r * 2: r * 2 + 2, c * 2: c * 2 + 2]
                mask = block == np.max(block)
                map_error[f, r * 2: r * 2 + 2, c * 2: c * 2 + 2] = mask * pooled_error[f, r, c]
    return map_error


def convolveBackward(image, filters, map_error):
    # filter change = sum over all positions of error x patch
    filter_change = np.zeros(filters.shape)
    size = len(image) - len(filters[0]) + 1
    for f in range(len(filters)):
        for r in range(size):
            for c in range(size):
                patch = image[r:r + len(filters[0]), c:c + len(filters[0])]
                e = map_error[f, r, c]
                filter_change[f] = filter_change[f] + e * patch
    return filter_change


def features(image, filters):
    # image -> convolution -> pooling -> one flat list for the dense part
    return maxPool(convolve(image, filters)).flatten()


class ConvolutionalNeuralNetwork:
    def __init__(self, filterCount, learningRate, filtersLearningRate):
        self.filters = np.random.normal(0.0, pow(9, -0.5), (filterCount, 3, 3))
        self.filtersLearningRate = filtersLearningRate
        # 28x28 image -> 26x26 feature maps -> 13x13 pooled maps
        self.dense = nn.NeuronalNetwork(filterCount * 13 * 13, 128, 10, learningRate)

    def train(self, train_image, train_label):
        # images may come flat (784) like in main.py, the filters need 28x28
        image = np.reshape(train_image, (28, 28))

        # forward
        kernel_map = convolve(image, self.filters)
        pooled = maxPool(kernel_map)
        flat = pooled.flatten()

        # dense part learns and returns the error for flat
        input_error = self.dense.train(flat, train_label)

        # backward through the CNN part
        pooled_error = input_error.reshape(pooled.shape)
        map_error = maxPoolBackward(kernel_map, pooled_error)
        filter_change = convolveBackward(image, self.filters, map_error)
        self.filters = self.filters + self.filtersLearningRate * filter_change

    def getOutput(self, test_image):
        image = np.reshape(test_image, (28, 28))
        return self.dense.getOutput(features(image, self.filters))

    def checkImage(self, test_image):
        return np.argmax(self.getOutput(test_image))

    def save(self, folder):
        self.dense.save(folder)
        np.save(os.path.join(folder, "filters.npy"), self.filters)

    def load(self, folder):
        # returns False if nothing has been saved in this folder yet
        path = os.path.join(folder, "filters.npy")
        if not os.path.exists(path) or not self.dense.load(folder):
            return False
        self.filters = np.load(path)
        return True

import nn
import cnn
import numpy as np

# MNIST IDX files: images have a 16-byte header, labels an 8-byte header
with open("train-images.idx3-ubyte", "rb") as f:
    train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
with open("train-labels.idx1-ubyte", "rb") as f:
    train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
with open("t10k-images.idx3-ubyte", "rb") as f:
    test_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
with open("t10k-labels.idx1-ubyte", "rb") as f:
    test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

train_images = train_images/255
test_images = test_images/255

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

model = input("Which model? (nn / cnn) ").strip().lower()
if model == "nn":
    netty = nn.NeuronalNetwork(784,128,10,0.00067)
    folder = "weights/nn"
elif model == "cnn":
    netty = cnn.ConvolutionalNeuralNetwork(8, 0.1, 0.025)
    folder = "weights/cnn"
else:
    raise SystemExit("Unknown model: " + model)

if netty.load(folder):
    print("loaded saved weights from", folder)
else:
    print("no saved weights, starting from random weights")

epochs = int(input("How many epochs? "))
for epoch in range(epochs):
    print("epoch", epoch + 1, "of", epochs)
    for i in range(len(train_images)):
        netty.train(train_images[i], train_labels[i])


correct = 0
for i in range(len(test_images)):
    guess = netty.checkImage(test_images[i])
    if guess == test_labels[i]:
        correct = correct + 1

print("correct:", correct, "of", len(test_images))
print("accuracy:", correct / len(test_images) * 100, "%")

netty.save(folder)
# AI-NumberDetector

A neural network that recognises handwritten digits (0–9), built from scratch with only numpy, without any machine learning library. It's a learning project: the goal is to understand how training, backpropagation and CNNs work by writing them by hand.

The network is trained on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset (60,000 training and 10,000 test images, 28×28 pixels).

## Features

- **Fully connected network** (`nn.py`): 784 inputs → 128 hidden neurons → 10 outputs, sigmoid activation, trained with backpropagation and gradient descent. Reaches about **96.8 %** accuracy on the 10,000 test images.
- **Saving and loading** of trained weights, so training doesn't start from scratch every time.
- **Drawing frontend** (`frontend.py`): draw a digit with the mouse and watch the network guess it live, including the 28×28 image it actually sees and the values of all 10 output neurons.
- **CNN, work in progress** (`cnn_playground.py`): convolution with 8 filters, max pooling and the backward pass for both. With random filters and only the dense part trained on 2,000 images, it reaches 86.8 % on 500 test images (the plain network: about 84 % on the same data).

## Getting started

Requirements: Python 3 with numpy and matplotlib.

```
pip install numpy matplotlib
```

The MNIST files are included in the repository.

**Train and test the network:**

```
python main.py
```

It asks which model to use (`nn` or `cnn`) and how many epochs to train. Entering `0` epochs only tests the saved weights. After the run, the weights are saved to `weights/nn/`.

**Draw your own digits:**

```
python frontend.py
```

This needs trained weights in `weights/nn/` (included).

## Project structure

| File | Contents |
| --- | --- |
| `nn.py` | the fully connected network: `train`, `getOutput`, `checkImage`, `save`, `load` |
| `main.py` | loads MNIST, trains for the chosen number of epochs, tests, saves the weights |
| `frontend.py` | drawing window with live prediction |
| `cnn_playground.py` | CNN building blocks: convolution, max pooling and their backward passes |
| `cnn.py` | the finished CNN (in progress) |
| `weights/nn/` | trained weights of the fully connected network |

## How it works (short version)

Each training step has four parts:

1. **Forward pass:** the image goes through the network and produces 10 output values.
2. **Error:** the outputs are compared with the correct answer (0.99 for the right digit, 0.01 for the others).
3. **Backpropagation:** the error is passed back through the weights to find out how much each weight contributed.
4. **Update:** every weight moves a small step in the direction that reduces the error.

The CNN adds small filters in front of this network. They slide over the image and look for patterns like strokes and edges, wherever those appear.

## Next steps

- Let the CNN filters learn (backprop all the way to the filters)
- Move the CNN into `cnn.py` and connect it to `main.py` and `frontend.py`
- Replace the Python loops with vectorised numpy operations for speed
- Compare with a PyTorch version on the GPU

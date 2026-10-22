# AI-NumberDetector

A neural network built from scratch with numpy that recognises handwritten digits (MNIST).

- `nn.py`: the `NeuronalNetwork` class (784 → 128 → 10, sigmoid, `train`, `checkImage`, `save` / `load`)
- `cnn.py`: the CNN: `convolve`, `maxPool`, `maxPoolBackward`, `convolveBackward`, `features` + class `ConvolutionalNeuralNetwork(filterCount, learningRate, filtersLearningRate)` with the same methods as the old network (accepts flat 784 or 28×28 images)
- `cnn_playground.py`: my CNN workspace (still has its own copies of the four functions)
- `main.py`: asks for the model (nn / cnn), loads its weights, trains for the chosen number of epochs, tests on 10,000 images, saves
- `frontend.py`: draw a digit and let the network guess it (`MODEL` at the top picks the network)
- `weights/nn/`: saved weights of the plain network (~96.8 % test accuracy); `weights/cnn/`: the CNN after 1 epoch (95.93 %)
- Any model plugged into `main.py` / `frontend.py` needs `train`, `getOutput`, `checkImage`, `save(folder)` and `load(folder)`

## How to work with me (the user)

I'm building this project to learn how neural networks work.

- **Teach, don't solve.** Explain concepts and ask guiding questions so I write the logic myself. Don't write AI logic for me unless I explicitly ask.
- **Small steps.** One question at a time. Start with small examples with concrete numbers (e.g. 3 pixels, 2 neurons), and only then generalise to numpy and matrices. If I say a step is too big, break it down further.
- **Mechanical changes are fine.** You can do renames, move code into a method or tidy up for me when I ask.
- **Language:** I switch between German and English. Answer in the language of my latest message.
- **Saving:** I sometimes forget to save files. Check the file on disk and point out unsaved changes.

## Status / next steps

- Done: save/load weights, epochs, drawing frontend, loss functions (MSE) explained
- Done: CNN works end to end (8 filters 3×3 → max pool → dense 1352 → 128 → 10), filters learn
  - `nn.py` `train` returns `input_error`, and now uses the textbook `hidden_error = np.dot(self.weight_ho.T, a)` (gradient check: matches the numerical gradient). Side effect: smaller steps, so the old network may need a higher learning rate when retrained
  - Settings: dense learning rate 0.5, filter learning rate 0.05
  - 1 epoch (~45 min with loops) → 95.93 % on 10,000 test images (old network after 1 epoch: ~95 %)
  - Frontend test: my own 1 (with hook) and 9 are still not recognised, an "American" 1 is. Cause: handwriting style missing in MNIST. Ideas: data augmentation, training on my own drawings
- Goal: 99 %. Plan, one change at a time and measured: 1. vectorise, 2. ReLU after the convolution, 3. softmax + cross-entropy, 4. more filters (32), 5. several epochs with decreasing learning rate, 6. maybe a second conv layer
- Now: vectorising, in `cnn.py`. So far: the idea of the patches matrix (im2col): all 3×3 patches as rows, (9, 9) for the 5×5 example, (676, 9) for MNIST. Next: build it with numpy, then convolution = one `np.dot` with the filters as (8, 9), check it against the loop version
- Later: maybe compare with PyTorch on the RTX 5070 (gaming PC)
- Study notes on the first network: https://claude.ai/code/artifact/0df8eb4a-7cc0-47e3-b975-684f6fff3228
- Study notes on the CNN so far: https://claude.ai/artifact/LNjn9mCagKpDN8dVKrjePV

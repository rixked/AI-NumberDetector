# AI-NumberDetector

A neural network built from scratch with numpy that recognises handwritten digits (MNIST).

- `nn.py`: the `NeuronalNetwork` class (784 → 128 → 10, sigmoid, `train`, `checkImage`, `save` / `load`)
- `cnn.py`: the CNN, which I'm writing myself (empty so far)
- `main.py`: asks for the model (nn / cnn), loads its weights, trains for the chosen number of epochs, tests on 10,000 images, saves
- `frontend.py`: draw a digit and let the network guess it (`MODEL` at the top picks the network)
- `weights/nn/`: saved weights of the plain network (~96.8 % test accuracy); the CNN will save to `weights/cnn/`
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
- Now: learning CNNs step by step in `cnn_playground.py` (will move to `cnn.py` once it works)
  - Done: `convolve` (8 random 3×3 filters → 8×26×26), `maxPool` (→ 8×13×13), `features` (flatten → 1352), dense part = `nn.NeuronalNetwork(1352, 128, 10, 0.1)`
  - Done: `maxPoolBackward` (error only to the max of each 2×2 block, via a mask), `convolveBackward` (filter change = sum over all positions of error × patch)
  - Result so far: random filters, only the dense part trained on 2,000 images → 86.8 % on 500 test images (old network on the same data: ~84 %)
  - Next: `train` in `nn.py` must return `input_error = np.dot(self.weight_ih.T, b)`, computed BEFORE `weight_ih` is updated. Then chain it: input_error → reshape (8, 13, 13) → `maxPoolBackward` → `convolveBackward` → `filters += learningRate * filter_change`
  - Side note: the `hidden_error` line uses `output_error`; textbook-correct would be `np.dot(self.weight_ho.T, a)`. Could be tried and compared later
- Later: vectorise the loops (~50–100× faster), maybe compare with PyTorch on the RTX 5070
- Study notes on the first network: https://claude.ai/code/artifact/0df8eb4a-7cc0-47e3-b975-684f6fff3228
- Study notes on the CNN so far: https://claude.ai/artifact/LNjn9mCagKpDN8dVKrjePV

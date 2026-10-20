# AI-NumberDetector

A neural network built from scratch with numpy that recognises handwritten digits (MNIST).

- `nn.py`: the `NeuronalNetwork` class (784 → 128 → 10, sigmoid, `train`, `checkImage`)
- `main.py`: loads MNIST, scales pixels to 0–1, trains on 60,000 images, tests on 10,000
- MNIST files (`*.idx3-ubyte`, `*.idx1-ubyte`) are not in git and have to be downloaded separately
- Current result: about 95 % test accuracy with learning rate 0.1 after one epoch

## How to work with me (the user)

I'm building this project to learn how neural networks work.

- **Teach, don't solve.** Explain concepts and ask guiding questions so I write the logic myself. Don't write AI logic for me unless I explicitly ask.
- **Small steps.** One question at a time. Start with small examples with concrete numbers (e.g. 3 pixels, 2 neurons), and only then generalise to numpy and matrices. If I say a step is too big, break it down further.
- **Mechanical changes are fine.** You can do renames, move code into a method or tidy up for me when I ask.
- **Language:** I switch between German and English. Answer in the language of my latest message.
- **Saving:** I sometimes forget to save files. Check the file on disk and point out unsaved changes.

## Status / next steps

- Next: save the weights with `np.save` / `np.load` so training isn't needed every time
- Ideas after that: multiple epochs, comparing learning rates, varying the number of hidden neurons
- Study notes on everything so far: https://claude.ai/code/artifact/0df8eb4a-7cc0-47e3-b975-684f6fff3228

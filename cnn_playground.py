import numpy as np
import nn

SHOW_PLOT = False   # True: show the filters, feature maps and pooled maps


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
    map_error = np.zeros(kernel_map.shape)
    poolSize = len(kernel_map[0]) // 2
    for f in range(len(kernel_map)):
        for r in range(poolSize):
            for c in range(poolSize):
                block = kernel_map[f, r * 2: r * 2 + 2, c * 2: c * 2 + 2]
                mask = block == np.max(block)
                map_error[f, r * 2: r * 2 + 2, c * 2: c * 2 + 2] = mask * pooled_error[f,r,c]
    return map_error

def convolveBackward(image, filters, map_error):
    filter_change = np.zeros(filters.shape)
    size = len(image) - len(filters[0]) + 1
    for f in range(len(filters)):
        for r in range(size):
            for c in range(size):
                patch = image[r:r + len(filters[0]), c:c + len(filters[0])]
                e = map_error[f,r,c]
                filter_change[f] = filter_change[f] + e * patch
    return filter_change

def features(image, filters):
    # image -> convolution -> pooling -> one flat list for the dense part
    return maxPool(convolve(image, filters)).flatten()


# MNIST training images as 28x28 arrays (pixels 0-1) and their labels
with open("train-images.idx3-ubyte", "rb") as f:
    train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28) / 255
with open("train-labels.idx1-ubyte", "rb") as f:
    train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

# MNIST test images (never used for training) and their labels
with open("t10k-images.idx3-ubyte", "rb") as f:
    test_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28) / 255
with open("t10k-labels.idx1-ubyte", "rb") as f:
    test_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

filters = np.random.normal(0.0, pow(9, -0.5), (8, 3, 3))

# the small 5x5 example from the start: arr = np.array([[0, 0, 1, 0, 0], ...])
arr = train_images[0]

kernel_map = convolve(arr, filters)
pooled = maxPool(kernel_map)
flat = features(arr, filters)
print(filters.shape, kernel_map.shape, pooled.shape, flat.shape)

dense = nn.NeuronalNetwork(len(flat), 128, 10, 0.1)
output = dense.getOutput(flat)
print(output)
print("guess:", np.argmax(output), "  correct:", train_labels[0])

for i in range(2000):
    dense.train(features(train_images[i],filters), train_labels[i])
correct = 0
for i in range(500):
    guess = dense.checkImage(features(test_images[i],filters))
    if guess == test_labels[i]:
        correct = correct + 1

print("correct:", correct, "of 500")
print("accuracy:", correct / 500 * 100, "%")


    
if SHOW_PLOT:
    # one column per filter: the filter itself, its feature map, its pooled map
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(filters) + 1, figsize=(16, 6))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].imshow(arr, cmap="gray")
    axes[0, 0].set_title("image")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "feature map\n" + str(kernel_map.shape[1:]), ha="center", va="center")
    axes[2, 0].text(0.5, 0.5, "pooled\n" + str(pooled.shape[1:]), ha="center", va="center")

    for f in range(len(filters)):
        axes[0, f + 1].imshow(filters[f], cmap="gray")
        axes[0, f + 1].set_title("filter " + str(f))
        axes[1, f + 1].imshow(kernel_map[f], cmap="gray")
        axes[2, f + 1].imshow(pooled[f], cmap="gray")

    plt.tight_layout()
    plt.show()

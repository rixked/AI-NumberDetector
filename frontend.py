import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

SIZE = 280   # drawing canvas in pixels (10x MNIST)
BRUSH = 12   # brush radius in canvas pixels

MODEL = "nn"   # switch to "cnn" once the CNN exists

if MODEL == "nn":
    netty = nn.NeuronalNetwork(784,128,10,0.1)
    folder = "weights/nn"
else:
    # TODO: create your CNN here once cnn.py is written (folder = "weights/cnn")
    raise SystemExit("The CNN doesn't exist yet.")

if not netty.load(folder):
    raise SystemExit("No saved weights in " + folder + ", run main.py first.")

canvas = np.zeros((SIZE, SIZE))
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
drawing = False
last = None


def paint(x, y):
    mask = (xx - x)**2 + (yy - y)**2 <= BRUSH**2
    canvas[mask] = 1.0


def resize(img, nh, nw):
    # shrink by averaging blocks of pixels
    h, w = img.shape
    ys = np.linspace(0, h, nh + 1).astype(int)
    xs = np.linspace(0, w, nw + 1).astype(int)
    out = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            block = img[ys[i]:max(ys[i+1], ys[i]+1), xs[j]:max(xs[j+1], xs[j]+1)]
            out[i, j] = block.mean()
    return out


def toMnist(img):
    # MNIST digits fit in a 20x20 box and are centred in 28x28 by their centre of mass
    rows = np.where(img.max(axis=1) > 0)[0]
    cols = np.where(img.max(axis=0) > 0)[0]
    if len(rows) == 0:
        return np.zeros((28, 28))
    crop = img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    h, w = crop.shape
    scale = 20 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = resize(crop, nh, nw)

    out = np.zeros((28, 28))
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top+nh, left:left+nw] = small

    total = out.sum()
    cy = (out.sum(axis=1) * np.arange(28)).sum() / total
    cx = (out.sum(axis=0) * np.arange(28)).sum() / total
    dy = int(np.clip(round(14 - cy), -4, 4))
    dx = int(np.clip(round(14 - cx), -4, 4))
    return np.roll(out, (dy, dx), axis=(0, 1))


fig = plt.figure(figsize=(12, 4.5))
fig.canvas.manager.set_window_title("AI-NumberDetector")
axDraw = fig.add_axes([0.03, 0.15, 0.3, 0.75])
axSeen = fig.add_axes([0.37, 0.15, 0.25, 0.75])
axBars = fig.add_axes([0.68, 0.15, 0.3, 0.75])
axClear = fig.add_axes([0.13, 0.03, 0.1, 0.07])

drawImg = axDraw.imshow(canvas, cmap="gray", vmin=0, vmax=1)
axDraw.set_title("draw a digit")
axDraw.set_xticks([])
axDraw.set_yticks([])

seenImg = axSeen.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
axSeen.set_title("what the network sees (28x28)")
axSeen.set_xticks([])
axSeen.set_yticks([])

bars = axBars.bar(range(10), np.zeros(10), color="gray")
axBars.set_xticks(range(10))
axBars.set_ylim(0, 1)
axBars.set_title("output neurons")

clearButton = Button(axClear, "Clear")


def predict():
    image = toMnist(canvas)
    seenImg.set_data(image)
    if image.sum() == 0:
        for bar in bars:
            bar.set_height(0)
            bar.set_color("gray")
        axBars.set_title("output neurons")
    else:
        output = netty.getOutput(image.reshape(784))
        guess = np.argmax(output)
        for digit, bar in enumerate(bars):
            bar.set_height(output[digit])
            bar.set_color("tab:green" if digit == guess else "gray")
        axBars.set_title("guess: " + str(guess))
    fig.canvas.draw_idle()


def onPress(event):
    global drawing, last
    if event.inaxes != axDraw:
        return
    drawing = True
    last = (event.xdata, event.ydata)
    paint(*last)
    drawImg.set_data(canvas)
    fig.canvas.draw_idle()


def onMove(event):
    global last
    if not drawing or event.inaxes != axDraw:
        return
    x, y = event.xdata, event.ydata
    lx, ly = last
    # fill the gap between two mouse positions so the line has no holes
    steps = int(np.hypot(x - lx, y - ly) / (BRUSH / 3)) + 1
    for t in np.linspace(0, 1, steps + 1)[1:]:
        paint(lx + (x - lx) * t, ly + (y - ly) * t)
    last = (x, y)
    drawImg.set_data(canvas)
    fig.canvas.draw_idle()


def onRelease(event):
    global drawing
    if drawing:
        drawing = False
        predict()


def onClear(event):
    canvas[:] = 0
    drawImg.set_data(canvas)
    predict()


fig.canvas.mpl_connect("button_press_event", onPress)
fig.canvas.mpl_connect("motion_notify_event", onMove)
fig.canvas.mpl_connect("button_release_event", onRelease)
clearButton.on_clicked(onClear)

plt.show()

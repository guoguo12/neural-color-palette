import re

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_X11_colors():
    colors = {}
    with open('rgb.txt') as f:
        for line in f.readlines():
            chunks = line.split()
            color_value = tuple(int(value) for value in chunks[:3])
            color_name = ' '.join(chunks[3:])

            # Only keep interesting colors
            # Every shade of 'grey' has a duplicate spelled with 'gray'
            if re.match('[a-z ]+$', color_name) and 'grey' not in color_name:
                colors[color_name] = color_value
    return colors


def predict(color, model=ResNet50(weights='imagenet')):
    x = np.broadcast_to(color, (1, 224, 224, 3)).astype(float)
    x = preprocess_input(x)
    return decode_predictions(model.predict(x), top=1)[0][0][1:]


def main():
    colors = get_X11_colors()
    sorted_colors = sorted(colors.items(), key=lambda x: x[1])

    fig, axes = plt.subplots(10, 10, figsize=(24, 16))
    for (ax, (color_name, color_value)) in tqdm(zip(axes.flatten(), sorted_colors)):
        ax.set_facecolor(tuple(float(value) / 255 for value in color_value))

        # Use basic heuristic to choose text color
        text_color = 'white' if sum(color_value) < 3 * 128 else 'black'

        predicted_name, prediction_posterior = predict(color_value)
        ax.text(0.5, 0.5, predicted_name, color=text_color, size=20,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(0.5, 0.25, prediction_posterior, color=text_color, size=12,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(color_name, size=12)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('result.png', pad_inches=0.05)

if __name__ == '__main__':
    main()

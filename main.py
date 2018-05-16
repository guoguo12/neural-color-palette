import re

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np


def get_x11_colors():
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


def predict(colors, model=ResNet50(weights='imagenet')):
    # Reshape colors from (N, 3) to (N, 1, 1, 3), and then to (N, 224, 224, 3)
    colors_array = np.array(colors)[:, np.newaxis, np.newaxis, :]
    colors_array = np.broadcast_to(colors_array,
                                   (len(colors), 224, 224, 3)).astype(float)

    x = preprocess_input(colors_array)
    predictions = decode_predictions(model.predict(x, verbose=1), top=1)
    return [(color, name, probability) \
            for (color, [(_, name, probability)]) in zip(colors, predictions)]


def plot_color(ax, color_value, x11_name, predicted_name, probability):
    ax.set_title(x11_name, size=12)
    ax.set_facecolor(tuple(float(value) / 255 for value in color_value))

    # Use heuristic to choose text color
    # Source: https://stackoverflow.com/a/3943023/7911713
    r, g, b = color_value
    text_color = 'k' if r * 0.299 + g * 0.587 + b * 0.116 > 186 else 'w'

    ax.text(0.5, 0.5, predicted_name,
            color=text_color, size=20,
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.2, '{:.3f}'.format(probability),
            color=text_color, size=12,
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_grid(results, x11_names):
    fig, axes = plt.subplots(24, 6, figsize=(28, 40))
    for (ax, x11_name, (color_value, predicted_name, probability)) \
        in zip(axes.flatten(), x11_names, results):
        plot_color(ax, color_value, x11_name, predicted_name, probability)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('grid.png', bbox_inches='tight', pad_inches=0.5)


def plot_uniques(results):
    unique_predicted_names = set(predicted_name \
                                 for (_, predicted_name, _) in results)

    fig, axes = plt.subplots(len(unique_predicted_names), 1, figsize=(12, 40))
    for (ax, predicted_name) in zip(axes, sorted(unique_predicted_names)):
        matching_results = [r for r in results if r[1] == predicted_name]
        color_value, _, probability = max(matching_results, key=lambda r: r[2])
        plot_color(ax, color_value, '', predicted_name, probability)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('uniques.png', bbox_inches='tight', pad_inches=0.5)


def main():
    colors = list(get_x11_colors().items())
    colors.sort(key=lambda x: x[1])  # Sort by RGB values
    x11_names, color_values = zip(*colors)

    results = predict(color_values)

    plot_grid(results, x11_names)
    plot_uniques(results)


if __name__ == '__main__':
    main()

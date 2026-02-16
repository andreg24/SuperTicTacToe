from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor

def plot_decision(output: Tensor | np.ndarray, log: bool = True):
    if isinstance(output, Tensor):
        im = output.cpu().detach().numpy()
    elif isinstance(output, Tensor):
        im = output
    im = im.squeeze()

    # check shape
    if im.shape == (81,):
        im = im.reshape(9, 9)
    elif im.shape != (9, 9):
        raise ValueError

    fig, ax = plt.subplots()

    img = ax.imshow(im, cmap='gray')

    # Add color scale
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=15)

    # Add thick 3×3 grid lines
    for x in range(0, 10, 3):
        ax.axvline(x - 0.5, color='red', linewidth=2)
        ax.axhline(x - 0.5, color='red', linewidth=2)

    # Optional fine grid for each cell
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)

    # Remove ticks and labels
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def plot_bar(r):
    """Plot bars of counts of rewards"""
    count = r['rewards_count']
    y1 = [0]*11
    y2 = [0]*11
    y0 = [0]*11
    y0[0]= count[0.0]

    values = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in range(len(values)):
        v = values[i]
        for k in count.keys():
            if abs(v-k)<0.0001:
                y1[i+1] = count[k]
            if abs(-v-k)<0.0001:
                y2[i+1] = count[k]
    x = list(range(11))
    plt.bar(x, y0, color='gray', label='Zero')
    plt.bar(x, y1, bottom=y0, color='r', label='Positive')
    plt.bar(x, y2, bottom=[a+b for a,b in zip(y0, y1)], color='b', label='Negative')
    plt.show()
    return y1, y2

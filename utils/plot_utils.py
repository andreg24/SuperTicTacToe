from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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


def extract_vlt(results: list) -> tuple[np.ndarray]:
    """From validation results extract np arrays for sequence of victories, losses, ties"""
    victories = []
    losses = []
    ties = []
    for res in results:
        v, l, t = res['results']
        victories.append(v)
        losses.append(l)
        ties.append(t)
    return np.array(victories), np.array(losses), np.array(ties)


# def plot_comp(a, b, c):
#     fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
#     a = extract_vlt(a)
#     b = extract_vlt(b)
#     c = extract_vlt(c)

#     axs[0].plot(a[0])
#     axs[1].plot(b[0])
#     axs[2].plot(c[0])

#     for i in range(3):
#         axs[i].axhline(39, color="black", linestyle="--", alpha=0.5)


def plot_comp(results: list, n=200, alpha=0.05, mode="v"):
    """plot """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    
    a, b, c = [], [], []
    
    for res in results:
        a.append(res[2])
        b.append(res[3])
        c.append(res[4])

    z = norm.ppf(1 - alpha / 2)
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    if mode == "v":
        idx = 0
    elif mode == "l":
        idx = 1
    elif mode == "t":
        idx = 2

    a = [extract_vlt(a[i])[idx] for i in range(len(a))]
    b = [extract_vlt(b[i])[idx] for i in range(len(b))]
    c = [extract_vlt(c[i])[idx] for i in range(len(c))]

    def add_plot(ax, series, title):
        p = np.asarray(series) / 100.0
        se = np.sqrt(p * (1 - p) / n)
        
        lower = np.clip((p - z * se) * 100, 0, 100)
        upper = np.clip((p + z * se) * 100, 0, 100)
        
        x = np.arange(len(series))
        ax.plot(x, series, label="win rate")
        ax.fill_between(x, lower, upper, alpha=0.2)
        ax.axhline(39, color="black", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_ylim(0, 100)

    for i in range(len(a)):
        add_plot(axs[0], a[i], "Agente 1 VS Agente 2")
        add_plot(axs[1], b[i], "Agent 1 VS Random")
        add_plot(axs[2], c[i], "Agente 2 VS Random")

    # plt.tight_layout()
    plt.show()


# def plot_val(r):
#     plt.figure(figsize=(16,6))
#     victories = []
#     losses = []
#     ties = []
#     for res in r:
#         v, l, t = res['results']
#         victories.append(v)
#         losses.append(l)
#         ties.append(t)
#     plt.plot(victories, label="victory")
#     plt.plot(losses, label="loss")
#     plt.plot(ties, label="tie")
#     plt.legend()


def plot_val(r, n=200, alpha=0.05):
    # 95% uses 1.96; if you want other alpha without scipy, you’d need a z-table.
    if abs(alpha - 0.05) < 1e-12:
        z = 1.96
    else:
        raise ValueError("For alpha != 0.05, compute z via a normal quantile (e.g., scipy.stats.norm.ppf).")

    victories, losses, ties = [], [], []
    for res in r:
        v, l, t = res['results']  # percentages 0..100
        victories.append(v)
        losses.append(l)
        ties.append(t)

    plt.figure(figsize=(16,6))

    def add_ci(series, label):
        p = np.asarray(series) / 100.0
        se = np.sqrt(p * (1 - p) / n)
        lower = np.clip((p - z * se) * 100, 0, 100)
        upper = np.clip((p + z * se) * 100, 0, 100)

        x = np.arange(len(series))
        plt.plot(x, series, label=label)
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.axhline(39, color="black", linestyle="--", alpha=0.5)

    add_ci(victories, "victory")
    add_ci(losses, "loss")
    add_ci(ties, "tie")

    plt.legend()
    plt.ylim(0, 100)
    plt.show()

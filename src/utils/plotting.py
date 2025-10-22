import json
import math
import os
import re
import pickle
import matplotlib.pyplot as plt
from typing import Any, Generator, Sequence

from utils import get_logger

logger = get_logger(__name__)


def smooth(scalars: list[float]) -> list[float]:
    """EMA implementation according to TensorBoard.

    Args:
        scalars: The scalars to smooth.

    Returns:
        The smoothed scalars.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot(values: dict[str, tuple[list[int], list[float]]], output_dir: str, y_label="metrics", smoothed: bool = True):
    """Plots curves and saves the image.

    Args:
        values: The values to plot.
        output_dir: The output directory.
        y_label: The y-axis label.
        smoothed: Whether to smooth the curves.
    """
    plt.switch_backend("agg")
    plt.figure()
    colors = plt.cm.get_cmap("tab10", len(values))
    for i, key in enumerate(values):
        if smoothed:
            plt.plot(*values[key], color=colors(i), alpha=0.4, label=f"{key} original")
            plt.plot(values[key][0], smooth(values[key][1]), color=colors(i), label=f"{key} smoothed")
        else:
            plt.plot(*values[key], color=colors(i), alpha=0.4, label=f"{key}")
    plt.xlabel("step")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{y_label}.png"), format="png", dpi=100)


def plot_trainer_state(
    trainer_state: dict[str, Any], keys: list[str], output_dir: str, y_label="metrics", smoothed: bool = False
):
    """Plots curves from the trainer state.

    Args:
        trainer_state: The trainer state.
        keys: The keys to plot.
        output_dir: The output directory.
        y_label: The y-axis label.
        smoothed: Whether to smooth the curves.
    """
    keys.extend([f"eval_{k}" for k in keys])
    values = {}
    for key in keys:
        steps, metrics = [], []
        for i in range(len(trainer_state["log_history"])):
            if key in trainer_state["log_history"][i]:
                steps.append(trainer_state["log_history"][i]["step"])
                metrics.append(trainer_state["log_history"][i][key])
        if len(metrics) == 0:
            continue
        values[key] = (steps, metrics)
    plot(values, output_dir, y_label, smoothed)

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

from pusht619.core import NUM_FACES


"""Visualize the gradient of the cost with respect to the action.

python scripts/main_visualize_grad_c.py --backward-dir logs/10__16:33:09__n-envs:2__lr:0.05__fixed-spawn__fixed-target__single-step/backward
"""

ACTION_DIM = 6


def plot_line(iterations, env_action_grads, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for face_idx in range(NUM_FACES):
        raw_grads = env_action_grads[:, face_idx]
        ax.plot(iterations, raw_grads, label=f"Face {face_idx}", alpha=0.4)
        if len(iterations) >= 5:
            smoothed = np.convolve(raw_grads, np.ones(5) / 5, mode="valid")
            ax.plot(iterations[2:-2], smoothed, linewidth=2, color=ax.lines[-1].get_color())
    ax.set_title("Line Plot (Raw & Smoothed)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_heatmap(iterations, env_action_grads, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        env_action_grads.T,
        aspect="auto",
        cmap="coolwarm",
        extent=[iterations[0], iterations[-1], NUM_FACES - 0.5, -0.5],
    )
    ax.set_title("Heatmap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Face Index")
    ax.set_yticks(range(NUM_FACES))
    fig.colorbar(im, ax=ax, label="Gradient Value")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_cumulative(iterations, env_action_grads, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative_grads = np.cumsum(env_action_grads, axis=0)
    for face_idx in range(NUM_FACES):
        ax.plot(iterations, cumulative_grads[:, face_idx], label=f"Face {face_idx}", linewidth=2)
    ax.set_title("Cumulative Gradient")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Gradient Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_stacked(iterations, env_action_grads, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom_pos = np.zeros(len(iterations))
    bottom_neg = np.zeros(len(iterations))

    for face_idx in range(NUM_FACES):
        grads = env_action_grads[:, face_idx]
        pos_grads = np.maximum(grads, 0)
        neg_grads = np.minimum(grads, 0)

        p = ax.bar(iterations, pos_grads, bottom=bottom_pos, label=f"Face {face_idx}")
        ax.bar(iterations, neg_grads, bottom=bottom_neg, color=p[0].get_facecolor())

        bottom_pos += pos_grads
        bottom_neg += neg_grads

    ax.set_title("Stacked Bar Chart")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_violin(iterations, env_action_grads, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [env_action_grads[:, i] for i in range(NUM_FACES)]
    ax.violinplot(data, positions=range(NUM_FACES), showmeans=True)
    ax.set_title("Gradient Distribution per Face")
    ax.set_xlabel("Face Index")
    ax.set_ylabel("Gradient Value")
    ax.set_xticks(range(NUM_FACES))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_scatter(iterations, env_action_grads, env_costs, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    grad_magnitudes = np.linalg.norm(env_action_grads, axis=1)
    sc = ax.scatter(env_costs, grad_magnitudes, c=iterations, cmap="viridis", alpha=0.7)
    ax.set_title("Gradient Magnitude vs. Mean Cost")
    ax.set_xlabel("Mean Cost")
    ax.set_ylabel("Gradient Magnitude (L2 Norm)")
    fig.colorbar(sc, ax=ax, label="Iteration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize grad_c_face from backward pass logs.")
    parser.add_argument(
        "--backward-dir", type=str, required=True, help="Path to the backward logs directory containing .npz files"
    )
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum number of iterations to visualize")
    args = parser.parse_args()

    backward_dir = Path(args.backward_dir)
    if not backward_dir.exists() or not backward_dir.is_dir():
        print(f"Error: Directory {backward_dir} does not exist.")
        return

    npz_files = []
    for f in backward_dir.glob("*.npz"):
        match = re.match(r"(\d+)\.npz", f.name)
        if match:
            npz_files.append((int(match.group(1)), f))

    npz_files.sort(key=lambda x: x[0])

    if args.max_iterations is not None:
        npz_files = npz_files[: args.max_iterations]

    if not npz_files:
        print(f"No .npz files found in {backward_dir}")
        return

    iterations = [x[0] for x in npz_files]

    all_grad_c_face = []
    all_costs = []
    for _, f in npz_files:
        data = np.load(f)
        all_grad_c_face.append(data["grad_c_face"])
        all_costs.append(np.mean(data["costs_per_env"], axis=0))

    all_grad_c_face = np.stack(all_grad_c_face, axis=0)
    all_costs = np.stack(all_costs, axis=0)

    n_iterations, n_envs, action_dim = all_grad_c_face.shape
    n_actions = action_dim // ACTION_DIM

    print(f"Loaded data for {n_iterations} iterations, {n_envs} environments, {n_actions} actions per step.")

    for env_idx in range(n_envs):
        for action_idx in range(n_actions):
            start_idx = action_idx * ACTION_DIM
            end_idx = start_idx + NUM_FACES
            env_action_grads = all_grad_c_face[:, env_idx, start_idx:end_idx]
            env_costs = all_costs[:, env_idx]

            base_path = backward_dir.parent / f"grad_c_face_env_{env_idx}_action_{action_idx}"

            plot_line(iterations, env_action_grads, f"{base_path}_line.png")
            plot_heatmap(iterations, env_action_grads, f"{base_path}_heatmap.png")
            plot_cumulative(iterations, env_action_grads, f"{base_path}_cumulative.png")
            plot_stacked(iterations, env_action_grads, f"{base_path}_stacked.png")
            plot_violin(iterations, env_action_grads, f"{base_path}_violin.png")
            plot_scatter(iterations, env_action_grads, env_costs, f"{base_path}_scatter.png")

        print(f"Saved all 6 plots for environment {env_idx}")


if __name__ == "__main__":
    main()

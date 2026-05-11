import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

from pusht619.core import NUM_FACES


"""Visualize the gradient of the cost with respect to the action.

python scripts/main_visualize_grad_c.py --n-envs 4 --n-mc-envs 9 --backward-dir logs/10__23:37:59__n-envs:4__lr:0.05__fixed-spawn__fixed-target__single-step/backward
"""

ACTION_DIM = 6


def plot_gradients_for_face(ax, iterations, c_vals, grads, face_idx, max_grad):

    ax.plot(iterations, c_vals, label=f"Face {face_idx} Logit (c)", color="blue", linewidth=2)

    assert c_vals.shape == (len(iterations),), f"c_vals.shape: {c_vals.shape}"
    assert grads.shape == (len(iterations),), f"grads.shape: {grads.shape}"

    for i, (x, y, g) in enumerate(zip(iterations, c_vals, grads)):
        if abs(g) > 1e-4:
            direction = 1 if g > 0 else -1
            color = "red" if g > 0 else "green"

            arrow_length = 0.5 * (abs(g) / max_grad)

            ax.annotate(
                "",
                xy=(x, y + direction * arrow_length),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
            )


def plot_c_and_grads_by_face(
    iterations: np.ndarray,
    c_env: np.ndarray,
    c_env_perturbed: np.ndarray,
    c_env_cost: np.ndarray,
    c_env_perturbed_costs: np.ndarray,
    mc_grad_c: np.ndarray,
    n_mc_envs: int,
    save_path: str,
    x_perturbed: np.ndarray,
    env_chosen_face: np.ndarray,
):
    """Same as plot_c_and_grads but there are NUM_FACES subplots, one for each face. Each subplot only shows the
    gradients that are from the randomly drawn pertrubation offsets that correspond to the particular face.
    """
    n_iters = len(iterations)
    assert c_env.shape == (
        n_iters,
        ACTION_DIM,
    )
    assert c_env_cost.shape == (n_iters,)
    assert c_env_perturbed.shape == (n_iters, n_mc_envs, ACTION_DIM)
    assert c_env_perturbed_costs.shape == (n_iters, n_mc_envs)
    c_env = c_env[:, :NUM_FACES]
    mc_grad_c = mc_grad_c[:, :NUM_FACES]
    c_env_perturbed = c_env_perturbed[:, :, :NUM_FACES]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(NUM_FACES, NUM_FACES + 1)
    # gs = fig.add_gridspec(NUM_FACES + 1, NUM_FACES + 1)

    axes = np.empty((NUM_FACES, NUM_FACES + 1), dtype=object)
    for i in range(NUM_FACES):
        for j in range(NUM_FACES + 1):
            if i == 0 and j == 0:
                axes[i, j] = fig.add_subplot(gs[i, j])
            else:
                axes[i, j] = fig.add_subplot(gs[i, j], sharex=axes[0, 0], sharey=axes[0, 0])
            axes[i, j].grid(True, alpha=0.3)
            plt.setp(axes[i, j].get_xticklabels(), visible=False)
            if j > 0:
                plt.setp(axes[i, j].get_yticklabels(), visible=False)

    axes[0, 0].set_title("Combined Gradients", fontsize=10)
    axes[0, 1].set_title("Gradients from Environments\nwhere Face 0 is Chosen Action", fontsize=10)
    axes[0, 2].set_title("Gradients from Environments\nwhere Face 1 is Chosen Action", fontsize=10)
    axes[0, 3].set_title("Gradients from Environments\nwhere Face 2 is Chosen Action", fontsize=10)
    axes[0, 4].set_title("Gradients from Environments\nwhere Face 3 is Chosen Action", fontsize=10)

    # Show the total gradient for each face in the first column
    max_grad = np.max(np.abs(mc_grad_c))
    for face_idx in range(NUM_FACES):
        ax = axes[face_idx, 0]
        c_vals = c_env[:, face_idx]
        grads = mc_grad_c[:, face_idx]
        plot_gradients_for_face(ax, iterations, c_vals, grads, face_idx, max_grad)
        ax.set_ylabel(f"Face {face_idx}")
        ax.grid(True, alpha=0.3)

    #
    for i in range(NUM_FACES):
        mean_grads = np.zeros((n_iters, NUM_FACES))

        for it_idx in range(n_iters):
            # Get indices of rows where the chosen face is i
            c_perturb_matching_idxs = np.where(np.argmax(x_perturbed[it_idx, :, :NUM_FACES], axis=1) == i)[0]

            if len(c_perturb_matching_idxs) == 0:
                continue

            for idx in c_perturb_matching_idxs:
                cost_diff = c_env_perturbed_costs[it_idx, idx] - c_env_cost[it_idx]
                c_delta = c_env_perturbed[it_idx, idx] - c_env[it_idx]
                mean_grads[it_idx, :] += cost_diff * c_delta
            mean_grads[it_idx, :] /= len(c_perturb_matching_idxs)

        for j in range(NUM_FACES):
            ax = axes[j, i + 1]
            c_vals = c_env[:, j]
            mean_grads_face = mean_grads[:, j]
            plot_gradients_for_face(ax, iterations, c_vals, mean_grads_face, face_idx, max_grad)

    # ax_bottom = fig.add_subplot(gs[NUM_FACES, :], sharex=axes[0, 0])
    # ax_bottom.scatter(iterations, env_chosen_face, color="purple", marker="o")
    # ax_bottom.plot(iterations, env_chosen_face, color="purple", alpha=0.3)
    # ax_bottom.set_ylim(-0.25, NUM_FACES + 0.25)
    # ax_bottom.set_yticks(range(NUM_FACES))
    # ax_bottom.set_ylabel("Chosen Face")
    # ax_bottom.set_xlabel("Iteration")
    # ax_bottom.grid(True, alpha=0.3)

    fig.suptitle("Face Logits (c) and Gradients over Iterations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot to {save_path}")
    return


def plot_c_and_grads(iterations, c, grad_c, env_chosen_face, save_path):
    fig, axes = plt.subplots(NUM_FACES + 1, 1, figsize=(12, 10), sharex=True)
    c = c[:, :NUM_FACES]

    min_c = np.min(c)
    max_c = np.max(c)
    c_range = max_c - min_c
    y_min = min_c - 0.2 * c_range
    y_max = max_c + 0.2 * c_range
    max_grad = np.max(np.abs(grad_c))

    for face_idx in range(NUM_FACES):
        ax = axes[face_idx]
        c_vals = c[:, face_idx]
        grads = grad_c[:, face_idx]
        plot_gradients_for_face(ax, iterations, c_vals, grads, face_idx, max_grad)

        ax.set_ylabel(f"Face {face_idx}")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    # Fifth subplot: chosen face
    ax = axes[NUM_FACES]
    ax.scatter(iterations, env_chosen_face, color="purple", marker="o")
    ax.plot(iterations, env_chosen_face, color="purple", alpha=0.3)
    ax.set_yticks(range(NUM_FACES))
    ax.set_ylabel("Chosen Face")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Face Logits (c) and Gradients over Iterations")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize grad_c_face from backward pass logs.")
    parser.add_argument(
        "--backward-dir", type=str, required=True, help="Path to the backward logs directory containing .npz files"
    )
    parser.add_argument("--max-iterations", type=int, default=250, help="Maximum number of iterations to visualize")
    parser.add_argument("--n-envs", type=int, required=True, help="Number of environments to visualize")
    parser.add_argument("--n-mc-envs", type=int, required=True, help="Number of MC environments to visualize")
    args = parser.parse_args()
    n_envs = args.n_envs
    n_mc_envs = args.n_mc_envs

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

    all_mc_grad_c = []
    all_c = []
    all_c_perturbed = []
    all_x_star = []
    all_eps_faces = []
    all_x_perturbed = []
    all_costs_per_env = []
    all_current_c_dists = []
    for _, f in npz_files:
        data = np.load(f)
        all_mc_grad_c.append(data["grad_c_face"])
        all_c.append(data["c"])
        all_c_perturbed.append(data["c_perturbed"])
        all_x_star.append(data["x_star"])
        all_eps_faces.append(data["eps_faces"])
        all_x_perturbed.append(data["x_perturbed"])
        all_costs_per_env.append(data["costs_per_env"])
        all_current_c_dists.append(data["current_c_dists"])

    all_mc_grad_c = np.stack(all_mc_grad_c, axis=0)
    all_c = np.stack(all_c, axis=0)
    all_c_perturbed = np.stack(all_c_perturbed, axis=0)
    all_x_star = np.stack(all_x_star, axis=0)
    all_eps_faces = np.stack(all_eps_faces, axis=0)
    all_x_perturbed = np.stack(all_x_perturbed, axis=0)
    all_costs_per_env = np.stack(all_costs_per_env, axis=0)
    all_current_c_dists = np.stack(all_current_c_dists, axis=0)

    n_iterations = len(iterations)
    assert all_mc_grad_c.shape == (n_iterations, n_envs, ACTION_DIM)
    assert all_c.shape == (n_iterations, n_envs, ACTION_DIM)
    assert all_c_perturbed.shape == (n_iterations, n_mc_envs, n_envs, ACTION_DIM)
    assert all_x_star.shape == (n_iterations, n_envs, ACTION_DIM)
    assert all_eps_faces.shape == (n_iterations, n_mc_envs, n_envs, ACTION_DIM)
    assert all_x_perturbed.shape == (n_iterations, n_mc_envs, n_envs, ACTION_DIM)
    assert all_costs_per_env.shape == (n_iterations, n_mc_envs, n_envs)
    assert all_current_c_dists.shape == (n_iterations, n_envs)

    print(f"Loaded data for {n_iterations} iterations, {n_envs} environments, {n_mc_envs} MC environments.")

    for env_idx in range(n_envs):
        env_grad_c_face = all_mc_grad_c[:, env_idx]
        env_c = all_c[:, env_idx]
        env_c_perturbed = all_c_perturbed[:, :, env_idx]
        env_x_star_face = all_x_star[:, env_idx]
        env_chosen_face = np.argmax(env_x_star_face[:, :NUM_FACES], axis=1)
        env_costs_per_env = all_costs_per_env[:, :, env_idx]
        env_current_c_dists = all_current_c_dists[:, env_idx]
        env_x_perturbed = all_x_perturbed[:, :, env_idx]
        base_path = backward_dir.parent / f"grad_c__env_{env_idx}"

        plot_c_and_grads(iterations, env_c, env_grad_c_face, env_chosen_face, f"{base_path}.png")
        plot_c_and_grads_by_face(
            iterations=iterations,
            c_env=env_c,
            c_env_perturbed=env_c_perturbed,
            c_env_cost=env_current_c_dists,
            c_env_perturbed_costs=env_costs_per_env,
            mc_grad_c=env_grad_c_face,
            n_mc_envs=n_mc_envs,
            save_path=f"{base_path}__per_face.png",
            x_perturbed=env_x_perturbed,
            env_chosen_face=env_chosen_face,
        )

    print(f"Saved plots for environment {env_idx}")


if __name__ == "__main__":
    main()

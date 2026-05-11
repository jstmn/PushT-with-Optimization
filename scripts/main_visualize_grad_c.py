import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

from pusht619.core import NUM_FACES


"""Visualize the gradient of the cost with respect to the action.

python scripts/main_visualize_grad_c.py --n-envs 4 --backward-dir logs/10__20:43:43__n-envs:4__lr:0.05__fixed-spawn__fixed-target__single-step/backward
"""

ACTION_DIM = 6


def plot_c_and_grads_by_face(iterations, c, c_perturbed, c_cost, c_perturbed_costs, save_path):
    """Same as plot_c_and_grads but there are NUM_FACES subplots, one for each face. Each subplot only shows the
    gradients that are from the randomly drawn pertrubation offsets that correspond to the particular face.
    """
    n_iters = len(iterations)
    fig, axes = plt.subplots(NUM_FACES + 1, NUM_FACES, figsize=(12, 10), sharex=True)


def plot_c_and_grads(iterations, save_path):
    fig, axes = plt.subplots(NUM_FACES + 1, 1, figsize=(12, 10), sharex=True)

    min_c = np.min(env_c)
    max_c = np.max(env_c)
    c_range = max_c - min_c if max_c > min_c else 1.0
    y_min = min_c - 0.25 * c_range
    y_max = max_c + 0.25 * c_range

    for face_idx in range(NUM_FACES):
        ax = axes[face_idx]
        c_vals = env_c[:, face_idx]
        grads = env_grad_c_face[:, face_idx]

        ax.plot(iterations, c_vals, label=f"Face {face_idx} Logit (c)", color="blue", linewidth=2)

        # Add arrows for gradients
        # Normalize gradients for arrow length
        max_grad = np.max(np.abs(grads)) if np.max(np.abs(grads)) > 0 else 1.0

        for i, (x, y, g) in enumerate(zip(iterations, c_vals, grads)):
            if abs(g) > 1e-4:
                direction = 1 if g > 0 else -1
                color = "red" if g > 0 else "green"

                # Scale arrow length slightly based on magnitude, but keep it visible
                arrow_length = 0.5 * (abs(g) / max_grad) * c_range
                if arrow_length < 0.05 * c_range:
                    arrow_length = 0.05 * c_range

                ax.annotate(
                    "",
                    xy=(x, y + direction * arrow_length),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
                )

        ax.set_ylabel(f"Face {face_idx}\nLogit")
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


def main():
    parser = argparse.ArgumentParser(description="Visualize grad_c_face from backward pass logs.")
    parser.add_argument(
        "--backward-dir", type=str, required=True, help="Path to the backward logs directory containing .npz files"
    )
    parser.add_argument("--max-iterations", type=int, default=250, help="Maximum number of iterations to visualize")
    parser.add_argument("--n-envs", type=int, required=True, help="Number of environments to visualize")
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
    all_c = []
    all_c_perturbed = []
    all_x_star = []
    all_eps_faces = []
    all_x_perturbed = []
    all_costs_per_env = []
    for _, f in npz_files:
        data = np.load(f)
        all_grad_c_face.append(data["grad_c_face"])
        all_c.append(data["c"])
        all_c_perturbed.append(data["c_perturbed"])
        all_x_star.append(data["x_star"])
        all_eps_faces.append(data["eps_faces"])
        all_x_perturbed.append(data["x_perturbed"])
        all_costs_per_env.append(data["costs_per_env"])

    all_grad_c_face = np.stack(all_grad_c_face, axis=0)
    all_c = np.stack(all_c, axis=0)
    all_c_perturbed = np.stack(all_c_perturbed, axis=0)
    all_x_star = np.stack(all_x_star, axis=0)
    all_eps_faces = np.stack(all_eps_faces, axis=0)
    all_x_perturbed = np.stack(all_x_perturbed, axis=0)
    all_costs_per_env = np.stack(all_costs_per_env, axis=0)

    n_iterations, n_envs, action_dim = all_grad_c_face.shape
    n_actions = action_dim // ACTION_DIM
    assert n_actions == 1
    assert n_envs == args.n_envs

    print(f"Loaded data for {n_iterations} iterations, {n_envs} environments, {n_actions} actions per step.")

    for env_idx in range(n_envs):
        env_grad_c_face = all_grad_c_face[:, env_idx]
        env_c = all_c[:, env_idx]
        env_c_perturbed = all_c_perturbed[:, env_idx]
        env_x_star_face = all_x_star[:, env_idx]
        env_chosen_face = np.argmax(env_x_star_face, axis=1)
        env_eps_faces = all_eps_faces[:, :, env_idx]
        env_x_perturbed = all_x_perturbed[:, :, env_idx]
        env_costs_per_env = all_costs_per_env[:, :, env_idx]

        base_path = backward_dir.parent / f"grad_c_face_env_{env_idx}"

        plot_c_and_grads(iterations, env_c_perturbed, env_grad_c_face, env_chosen_face, f"{base_path}_stacked_c.png")
        plot_c_and_grads_by_face(
            iterations,
            env_c,
            env_grad_c_face,
            env_chosen_face,
            env_eps_faces,
            env_x_perturbed,
            env_costs_per_env,
            f"{base_path}_by_face.png",
        )

    print(f"Saved plots for environment {env_idx}")


if __name__ == "__main__":
    main()

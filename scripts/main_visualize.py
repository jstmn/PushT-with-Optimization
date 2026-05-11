"""
This script visualizes the cost landscape for a single environment across different faces,
contact points, and angles. It creates a figure with 4 subplots (one for each face),
where each subplot is a 2D heatmap showing the difference in final distance
(final_distance - initial_distance) for a grid of 5x5 contact points and angles.


python scripts/main_visualize.py --iterations-dir logs/10__20:43:43__n-envs:4__lr:0.05__fixed-spawn__fixed-target__single-step/iterations
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from pusht619.core import (
    ANGLE_BOUNDS,
    CONTACT_POINT_BOUNDS,
    NUM_FACES,
    get_action_cost_matrix_cache_filename,
    PushTEnv,
    Action,
)
from pusht619.plotting_utils import draw_scene_visualization

N_SIM_STEPS = 25
N_GRID = 9

FACE_DESCRIPTIONS = ["Left side of stem", "Bottom of stem", "Right side of stem", "Top of top bar"]


def plot_for_env(t_pose, t_vel, target_pose, cp_vals, ang_vals, save_path, action_history=None):
    # Create the figure with 5 subplots
    fig, axes = plt.subplots(1, NUM_FACES + 1, figsize=(25, 5))

    cache_file = get_action_cost_matrix_cache_filename(t_pose, t_vel, target_pose, N_SIM_STEPS, N_GRID)

    if cache_file.exists():
        print(f"Loading cached cost matrices from {cache_file}")
        all_cost_matrices = np.load(cache_file)
    else:
        env = PushTEnv(
            nenvs=1,
            record_video=False,
            visualize=False,
            use_relative_coordinates=True,
            random_mode="fixed-spawn__fixed-target",
        )

        all_cost_matrices = []
        n_actions = NUM_FACES * N_GRID * N_GRID
        counter = tqdm(total=n_actions, desc="Computing cost matrices")
        for face_idx in range(NUM_FACES):
            cost_matrix = np.zeros((N_GRID, N_GRID))

            for i, cp in enumerate(cp_vals):
                for j, ang in enumerate(ang_vals):
                    # Reset to the exact same state
                    env.reset(
                        target_poses=np.array([target_pose]),
                        t_poses=np.array([t_pose]),
                        joint_velocities=np.array([t_vel]),
                    )

                    action = Action(
                        face=np.full((1, 1), face_idx, dtype=np.int32),
                        contact_point=np.full((1, 1), cp, dtype=np.float32),
                        angle=np.full((1, 1), ang, dtype=np.float32),
                    )
                    result = env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
                    t_distances = np.asarray(result.t_distances)

                    # Difference in final distance (final - initial)
                    # If negative, it means the distance decreased (which is good)
                    cost = t_distances[0, -1] - t_distances[0, 0]
                    cost_matrix[j, i] = cost
                    counter.update(1)

            all_cost_matrices.append(cost_matrix)

        counter.close()
        all_cost_matrices = np.array(all_cost_matrices)
        np.save(cache_file, all_cost_matrices)
        print(f"Saved cost matrices to cache: {cache_file}")

    # Find global min and max for consistent color scaling
    vmin = np.min(all_cost_matrices)
    vmax = np.max(all_cost_matrices)

    for face_idx in range(NUM_FACES):
        ax = axes[face_idx]
        # Plot heatmap
        im = ax.imshow(
            all_cost_matrices[face_idx],
            origin="lower",
            extent=[cp_vals[0], cp_vals[-1], ang_vals[0], ang_vals[-1]],
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Face {face_idx}: {FACE_DESCRIPTIONS[face_idx]}")
        ax.set_xlabel("Contact Point")
        ax.set_ylabel("Angle")

        # Overlay action history if provided
        if action_history is not None:
            # action_history is a list of (face, cp, ang) tuples
            history_for_face = [(idx, cp, ang) for idx, (f, cp, ang) in enumerate(action_history) if f == face_idx]
            if history_for_face:
                # Create a colormap
                cmap = plt.get_cmap("rainbow")
                # Normalize indices to [0, 1] range for the colormap
                max_idx = len(action_history) - 1
                norm = plt.Normalize(vmin=0, vmax=max_idx)

                # Plot each point with its specific color based on its global iteration index
                for idx, cp, ang in history_for_face:
                    color = cmap(norm(idx))
                    ax.scatter([cp], [ang], color=[color], s=30, edgecolors="white", zorder=5)
                    # Add iteration numbers
                    ax.annotate(
                        str(idx),
                        (cp, ang),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                        zorder=6,
                    )

        # Only show the colorbar on the rightmost subplot
        if face_idx == NUM_FACES - 1:
            fig.colorbar(im, ax=ax, label="Distance Diff (Final - Initial)")

    # Draw the scene in the last subplot
    draw_scene_visualization(axes[-1], t_pose, target_pose)

    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations-dir", type=str, default=None, help="Path to iterations directory")
    parser.add_argument("--n-steps-max", type=int, default=50)
    args = parser.parse_args()

    # We want 5 values for contact point and 5 values for angle
    cp_vals = np.linspace(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1], N_GRID)
    ang_vals = np.linspace(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]), N_GRID)

    if args.iterations_dir:
        iterations_dir = Path(args.iterations_dir)
        json_files = sorted(iterations_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {iterations_dir}")
            return

        # Read the first file to get n_envs
        with open(json_files[0], "r") as f:
            data0 = json.load(f)
        n_envs = len(data0["t_poses"])

        for env_idx in range(n_envs):
            print(f"Processing environment {env_idx}/{n_envs}...")

            # Extract history for this env
            t_pose_0 = np.array(data0["t_poses"][env_idx])
            t_vel_0 = np.array(data0["t_velocities"][env_idx])
            target_pose_0 = np.array(data0["target_poses"][env_idx])
            action_history = []
            valid = True

            for jf in json_files:
                with open(jf, "r") as f:
                    data = json.load(f)

                t_pose = np.array(data["t_poses"][env_idx])
                t_vel = np.array(data["t_velocities"][env_idx])
                target_pose = np.array(data["target_poses"][env_idx])

                if (
                    not np.allclose(t_pose, t_pose_0, atol=1e-4)
                    or not np.allclose(t_vel, t_vel_0, atol=1e-4)
                    or not np.allclose(target_pose, target_pose_0, atol=1e-4)
                ):
                    print(f"Warning: Env {env_idx} state changed during optimization! Skipping.")
                    valid = False
                    break

                x = np.array(data["x"][env_idx])
                face_weights = x[:NUM_FACES]
                face = np.argmax(face_weights)
                cp = x[NUM_FACES]
                ang = x[NUM_FACES + 1]
                action_history.append((face, cp, ang))
                if len(action_history) >= args.n_steps_max:
                    break

            if valid:
                save_path = iterations_dir.parent / f"cost_landscape_env_{env_idx}.png"
                plot_for_env(t_pose_0, t_vel_0, target_pose_0, cp_vals, ang_vals, save_path, action_history)

    else:
        # Default behavior: single plot for seed=0
        from pusht619.core import PushTEnv

        env = PushTEnv(
            nenvs=1,
            record_video=False,
            visualize=False,
            use_relative_coordinates=True,
            random_mode="fixed-spawn__fixed-target",
        )
        env.reset()
        t_pose = env.t_poses[0].copy()
        t_vel = env.t_velocities[0].copy()
        target_pose = env.target_poses[0].copy()
        save_path = Path("logs/cost_landscape.png")
        plot_for_env(t_pose, t_vel, target_pose, cp_vals, ang_vals, save_path)


if __name__ == "__main__":
    main()

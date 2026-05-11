from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES
from pusht619.core import get_t_and_target_corners, T_CORNERS, FACE_START_POINTS, FACE_END_POINTS


def plot_results(
    dist_change_hist,
    face_hist,
    cp_hist,
    ang_hist,
    n_envs,
    n_opt_steps,
    random_mode: str,
    m_rollouts: int,
    perturb_lambda: float,
    relative_coordinates: bool,
    save_filepath=None,
    save_filepath2=None,
    open_after_save=False,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"SurCo-prior  n_envs={n_envs}  "
        f"n_opt_steps={n_opt_steps}  M={m_rollouts}  λ={perturb_lambda}  "
        f"RANDOM_MODE={random_mode}  RELATIVE_COORDINATES={relative_coordinates}",
        fontweight="bold",
    )
    ax_cp, ax_ang = axes[0, 0], axes[0, 1]
    ax_face, ax_dist = axes[1, 0], axes[1, 1]
    n_envs_max = min(n_envs, 7)

    mean_dist_change = [float(np.mean(dists)) for dists in dist_change_hist]
    ax_dist.plot(mean_dist_change, color="black", linewidth=2.0, label="mean")
    for env_idx in range(n_envs_max):
        ax_dist.plot([dists[env_idx] for dists in dist_change_hist], label=f"env {env_idx}", alpha=0.8)
    ax_dist.legend()
    ax_dist.set_title("Distance Change Per Env during rollout (Final - Initial)")
    ax_dist.set_xlabel("Iteration")
    ax_dist.set_ylabel("Distance Change [cm]")
    ax_dist.grid(True, alpha=0.3)

    for env_idx in range(n_envs_max):
        x1 = np.arange(len(cp_hist))
        p1 = ax_cp.plot([cp[env_idx] for cp in cp_hist], label=f"env {env_idx}", alpha=0.5)
        ax_cp.scatter(x1, [cp[env_idx] for cp in cp_hist], color=p1[0].get_color())
        p2 = ax_ang.plot([a[env_idx] for a in ang_hist], label=f"env {env_idx}", alpha=0.5)
        ax_ang.scatter(x1, [a[env_idx] for a in ang_hist], color=p2[0].get_color())
        p3 = ax_face.plot([f[env_idx] for f in face_hist], marker=".", linestyle="-", label=f"env {env_idx}", alpha=0.5)
        ax_face.scatter(x1, [f[env_idx] for f in face_hist], color=p3[0].get_color())

    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
    ax_cp.axhline(lo_cp, color="gray", linestyle="--", linewidth=0.8)
    ax_cp.axhline(hi_cp, color="gray", linestyle="--", linewidth=0.8)
    ax_cp.legend()
    ax_cp.set_title("Contact Point")
    ax_cp.set_xlabel("Iteration")
    ax_cp.set_ylabel("contact_point")
    ax_cp.grid(True, alpha=0.3)

    ax_ang.axhline(lo_ang, color="gray", linestyle="--", linewidth=0.8)
    ax_ang.axhline(hi_ang, color="gray", linestyle="--", linewidth=0.8)
    ax_ang.legend()
    ax_ang.set_title("Angle")
    ax_ang.set_xlabel("Iteration")
    ax_ang.set_ylabel("angle [rad]")
    ax_ang.grid(True, alpha=0.3)

    ax_face.set_yticks(range(NUM_FACES))
    ax_face.legend()
    ax_face.set_title("Face Chosen (argmax)")
    ax_face.set_xlabel("Iteration")
    ax_face.set_ylabel("face index")
    ax_face.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    if save_filepath2 is not None:
        plt.savefig(save_filepath2, bbox_inches="tight")
        print(f"Saved plot to {save_filepath2}")
    plt.close()
    if open_after_save:
        print(f"xdg-open {save_filepath}")
        os.system(f"xdg-open {save_filepath}")


def plot_perturbation_hist(
    c: np.ndarray,  # (n_envs, ACTION_DIM)
    c_pert_ks: list,  # M x (n_envs, ACTION_DIM)
    x_ks: list,  # M x (n_envs, ACTION_DIM)
    x_star: np.ndarray,  # (n_envs, ACTION_DIM)
    n_envs: int,
    iteration: int,
) -> plt.Figure:
    c_np = np.asarray(c)
    c_pert_all = np.stack([np.asarray(ck) for ck in c_pert_ks])  # (M, n_envs, ACTION_DIM)
    x_all = np.stack([np.asarray(xk) for xk in x_ks])  # (M, n_envs, ACTION_DIM)
    x_star_np = np.asarray(x_star)

    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])

    n_rows = min(n_envs, 5)
    n_cols = NUM_FACES + 3  # face_0…face_N | argmax | cp | angle
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle(f"c perturbations — iter {iteration}", fontsize=9)

    # face_vals = c_pert_all[:, :n_rows, :NUM_FACES]
    # face_xlim = (float(face_vals.min()), float(face_vals.max()))
    face_xlim = (None, None)

    for row, env_i in enumerate(range(n_rows)):
        # ── face logit histograms ──────────────────────────────────────
        for fi in range(NUM_FACES):
            ax = axes[row, fi]
            ax.hist(c_pert_all[:, env_i, fi], bins=20, color="steelblue", alpha=0.7)
            ax.axvline(c_np[env_i, fi], color="red", linewidth=1.5, label="c")
            ax.set_xlim(face_xlim)
            if row == 0:
                ax.set_title(f"c_face_{fi}", fontsize=8)
            ax.set_ylabel(f"env {env_i}", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and fi == 0:
                ax.legend(fontsize=6)

        # ── argmax face bar chart ──────────────────────────────────────
        ax_face = axes[row, NUM_FACES]
        face_choices_env = np.argmax(x_all[:, env_i, :NUM_FACES], axis=-1)
        ax_face.bar(range(NUM_FACES), np.bincount(face_choices_env, minlength=NUM_FACES), color="steelblue", alpha=0.7)
        chosen = int(np.argmax(x_star_np[env_i, :NUM_FACES]))
        ax_face.axvline(chosen, color="red", linewidth=1.5, label="x_star")
        ax_face.set_xticks(range(NUM_FACES))
        if row == 0:
            ax_face.set_title("argmax face", fontsize=8)
        ax_face.tick_params(labelsize=6)
        if row == 0:
            ax_face.legend(fontsize=6)

        # ── contact_point histogram ────────────────────────────────────
        ax_cp = axes[row, NUM_FACES + 1]
        ax_cp.hist(c_pert_all[:, env_i, NUM_FACES], bins=20, color="steelblue", alpha=0.7)
        ax_cp.axvline(c_np[env_i, NUM_FACES], color="red", linewidth=1.5)
        ax_cp.set_xlim(lo_cp, hi_cp)
        if row == 0:
            ax_cp.set_title("c_contact_point", fontsize=8)
        ax_cp.tick_params(labelsize=6)

        # ── angle histogram ────────────────────────────────────────────
        ax_ang = axes[row, NUM_FACES + 2]
        ax_ang.hist(c_pert_all[:, env_i, NUM_FACES + 1], bins=20, color="steelblue", alpha=0.7)
        ax_ang.axvline(c_np[env_i, NUM_FACES + 1], color="red", linewidth=1.5)
        ax_ang.set_xlim(lo_ang, hi_ang)
        if row == 0:
            ax_ang.set_title("c_angle", fontsize=8)
        ax_ang.tick_params(labelsize=6)

    fig.tight_layout()
    return fig


def plot_network_output_hist(
    c: np.ndarray,  # (n_envs, ACTION_DIM)
    x_star: np.ndarray,  # (n_envs, ACTION_DIM)
    iteration: int,
) -> plt.Figure:
    """Histogram of raw network outputs across environments — same column layout as plot_perturbation_hist."""
    c_np = np.asarray(c)  # (n_envs, ACTION_DIM)
    x_np = np.asarray(x_star)  # (n_envs, ACTION_DIM)
    n_envs = c_np.shape[0]

    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])

    n_cols = NUM_FACES + 3  # face_0…face_N | argmax | cp | angle
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), squeeze=False)
    axes = axes[0]
    fig.suptitle(f"network output distribution — iter {iteration}", fontsize=9)

    face_xlim = (float(c_np[:, :NUM_FACES].min()), float(c_np[:, :NUM_FACES].max()))

    for fi in range(NUM_FACES):
        ax = axes[fi]
        ax.hist(c_np[:, fi], bins=20, color="steelblue", alpha=0.7)
        ax.set_xlim(face_xlim)
        ax.set_title(f"c_face_{fi}", fontsize=8)
        ax.tick_params(labelsize=6)

    ax_face = axes[NUM_FACES]
    face_choices = np.argmax(x_np[:, :NUM_FACES], axis=-1)
    ax_face.bar(range(NUM_FACES), np.bincount(face_choices, minlength=NUM_FACES), color="steelblue", alpha=0.7)
    ax_face.set_ylim(0, n_envs)
    ax_face.set_xticks(range(NUM_FACES))
    ax_face.set_title("chosen face", fontsize=8)
    ax_face.tick_params(labelsize=6)

    ax_cp = axes[NUM_FACES + 1]
    ax_cp.hist(c_np[:, NUM_FACES], bins=20, color="steelblue", alpha=0.7)
    ax_cp.set_xlim(lo_cp, hi_cp)
    ax_cp.set_title("c_contact_point", fontsize=8)
    ax_cp.tick_params(labelsize=6)

    ax_ang = axes[NUM_FACES + 2]
    ax_ang.hist(c_np[:, NUM_FACES + 1], bins=20, color="steelblue", alpha=0.7)
    ax_ang.set_xlim(lo_ang, hi_ang)
    ax_ang.set_title("c_angle", fontsize=8)
    ax_ang.tick_params(labelsize=6)

    for ax in axes:
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def draw_scene_visualization(ax_scene, t_pose, target_pose):
    """Draws the T block, target T block, and face labels on the given axes."""
    t_corners, target_corners = get_t_and_target_corners(np.array([t_pose]), np.array([target_pose]))

    # Plot target T (green)
    tgt_poly = plt.Polygon(
        target_corners[0], closed=True, fill=True, facecolor="green", alpha=0.3, edgecolor="green", linestyle="--"
    )
    ax_scene.add_patch(tgt_poly)

    # Plot T (orange)
    t_poly = plt.Polygon(t_corners[0], closed=True, fill=True, facecolor="orange", alpha=0.7, edgecolor="darkorange")
    ax_scene.add_patch(t_poly)

    # Add face labels to the T block
    for face_idx in range(NUM_FACES):
        # Find the indices of the start and end points in T_CORNERS
        start_idx = np.where(np.all(T_CORNERS == FACE_START_POINTS[face_idx], axis=1))[0][0]
        end_idx = np.where(np.all(T_CORNERS == FACE_END_POINTS[face_idx], axis=1))[0][0]
        p_start = t_corners[0, start_idx]
        p_end = t_corners[0, end_idx]
        midpoint = (p_start + p_end) / 2

        # Add a small offset outward for the text
        normal = np.array([-(p_end[1] - p_start[1]), p_end[0] - p_start[0]])
        normal = normal / np.linalg.norm(normal)
        text_pos = midpoint + normal * 0.02

        ax_scene.text(
            text_pos[0],
            text_pos[1],
            f"Face {face_idx}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

        # Draw a line highlighting the face
        ax_scene.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], "r-", linewidth=2)

    ax_scene.set_aspect("equal")
    ax_scene.set_title("Scene Visualization")
    ax_scene.set_xlabel("X")
    ax_scene.set_ylabel("Y")

    # Set limits based on the block positions
    all_pts = np.vstack([t_corners[0], target_corners[0]])
    min_x, min_y = np.min(all_pts, axis=0) - 0.1
    max_x, max_y = np.max(all_pts, axis=0) + 0.1
    ax_scene.set_xlim(min_x, max_x)
    ax_scene.set_ylim(min_y, max_y)


def plot_rollout(
    save_dir,
    t_poses_initial,
    t_poses_final,
    target_poses,
    pusher_initial_positions,
    save_filepath=None,
):
    """Plots the before and after for every T, pusher's initial position, and an arrow showing T movement."""
    from pusht619.core import get_t_and_target_corners
    import numpy as np

    n_envs = t_poses_initial.shape[0]

    # Check if there's a single target T (all target poses are the same)
    single_target = True
    if n_envs > 1:
        for i in range(1, n_envs):
            if not np.allclose(target_poses[0], target_poses[i]):
                single_target = False
                break

    if single_target:
        fig, ax = plt.subplots(figsize=(10, 10))
        axes = [ax]
        envs_to_plot = [list(range(min(n_envs, 9)))]
    else:
        n_plots = min(n_envs, 9)
        if n_plots == 1:
            rows, cols = 1, 1
            figsize = (10, 10)
        elif n_plots == 2:
            rows, cols = 1, 2
            figsize = (16, 8)
        elif n_plots <= 4:
            rows, cols = 2, 2
            figsize = (16, 16)
        else:
            rows, cols = 3, 3
            figsize = (20, 20)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        envs_to_plot = [[i] for i in range(n_plots)]

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

    for plot_idx, env_indices in enumerate(envs_to_plot):
        ax = axes[plot_idx]
        all_pts = []

        # Plot target T(s)
        if single_target:
            _, target_corners = get_t_and_target_corners(np.array([t_poses_initial[0]]), np.array([target_poses[0]]))
            tgt_poly = plt.Polygon(
                target_corners[0],
                closed=True,
                fill=True,
                facecolor="green",
                alpha=0.3,
                edgecolor="green",
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(tgt_poly)
            all_pts.append(target_corners[0])
        else:
            for i in env_indices:
                _, target_corners = get_t_and_target_corners(
                    np.array([t_poses_initial[i]]), np.array([target_poses[i]])
                )
                tgt_poly = plt.Polygon(
                    target_corners[0],
                    closed=True,
                    fill=True,
                    facecolor="green",
                    alpha=0.3,
                    edgecolor="green",
                    linestyle="--",
                    linewidth=2,
                )
                ax.add_patch(tgt_poly)
                all_pts.append(target_corners[0])

        for i in env_indices:
            # Initial T
            t_corners_initial, _ = get_t_and_target_corners(np.array([t_poses_initial[i]]), np.array([target_poses[i]]))
            t_poly_initial = plt.Polygon(
                t_corners_initial[0], closed=True, fill=True, facecolor="orange", alpha=0.3, edgecolor="darkorange"
            )
            ax.add_patch(t_poly_initial)
            all_pts.append(t_corners_initial[0])

            # Final T
            t_corners_final, _ = get_t_and_target_corners(np.array([t_poses_final[i]]), np.array([target_poses[i]]))
            t_poly_final = plt.Polygon(
                t_corners_final[0], closed=True, fill=True, facecolor="orange", alpha=0.8, edgecolor="darkorange"
            )
            ax.add_patch(t_poly_final)
            all_pts.append(t_corners_final[0])

            # Pusher initial position
            ax.scatter(
                pusher_initial_positions[i, 0],
                pusher_initial_positions[i, 1],
                color="blue",
                marker="o",
                s=50,
                label="Pusher Initial" if plot_idx == 0 and i == env_indices[0] else "",
            )
            all_pts.append(pusher_initial_positions[i : i + 1])

            # Arrow from initial to final T center
            ax.arrow(
                t_poses_initial[i, 0],
                t_poses_initial[i, 1],
                t_poses_final[i, 0] - t_poses_initial[i, 0],
                t_poses_final[i, 1] - t_poses_initial[i, 1],
                head_width=0.02,
                head_length=0.03,
                fc="black",
                ec="black",
                alpha=0.5,
                length_includes_head=True,
            )

        # Set limits based on all points
        if all_pts:
            all_pts_arr = np.vstack(all_pts)
            min_x, min_y = np.min(all_pts_arr, axis=0) - 0.1
            max_x, max_y = np.max(all_pts_arr, axis=0) + 0.1
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        ax.set_aspect("equal")
        if single_target:
            ax.set_title("Rollout Visualization (Before & After)")
        else:
            ax.set_title(f"Env {env_indices[0]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

    if not single_target:
        fig.suptitle("Rollout Visualization (Before & After)", fontsize=16)

    fig.tight_layout()
    if save_filepath is None:
        save_filepath = save_dir / "rollout_visualization.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved rollout plot to {save_filepath}")
    plt.close(fig)

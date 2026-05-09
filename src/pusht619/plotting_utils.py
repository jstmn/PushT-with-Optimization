from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES


def plot_results(
    save_dir,
    means,
    stds,
    dist_delta_hist,
    face_hist,
    cp_hist,
    ang_hist,
    n_envs,
    n_sim_steps,
    n_opt_steps,
    random_t_pose,
    m_rollouts: int,
    perturb_lambda: float,
    relative_coordinates: bool = False,
    random_means=None,
    random_stds=None,
    baseline_iters=None,
    save_filepath=None,
    save_filepath2=None,
    open_after_save=False,
):
    initial_mean_loss = means[0]
    x_iters = np.arange(len(means))
    has_random_baseline = random_means is not None and random_stds is not None
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  n_envs={n_envs}  "
        f"n_opt_steps={n_opt_steps}  M={m_rollouts}  λ={perturb_lambda}  "
        f"RANDOM_T_POSE={random_t_pose}  RELATIVE_COORDINATES={relative_coordinates}",
        fontweight="bold",
    )
    ax_mean, ax_std = axes[0, 0], axes[0, 1]
    ax_cp, ax_ang = axes[1, 0], axes[1, 1]
    ax_face, ax_delta = axes[2, 0], axes[2, 1]
    n_envs_max = min(n_envs, 7)

    ax_mean.axhline(float(initial_mean_loss), label="initial mean", color="black", linestyle="--")
    ax_mean.plot(x_iters, means, label="training mean", color="tab:red")
    ax_mean.legend()
    ax_mean.set_title("Mean Final Distance")
    ax_mean.set_xlabel("Iteration")
    ax_mean.set_ylabel("Distance [m]")
    ax_mean.grid(True, alpha=0.3)

    if has_random_baseline:
        bx = np.asarray(baseline_iters) if baseline_iters is not None else np.arange(len(random_means))
        ax_std.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax_std.plot(bx, random_means, label="mean % vs baseline", color="tab:green")
        ax_std.fill_between(
            bx,
            np.asarray(random_means) - np.asarray(random_stds),
            np.asarray(random_means) + np.asarray(random_stds),
            color="tab:green",
            alpha=0.2,
            label="± std across envs",
        )
        ax_std.legend()
        ax_std.set_title("% vs Center-Action Baseline")
        ax_std.set_xlabel("Iteration")
        ax_std.set_ylabel("% change (negative = better)")
    else:
        ax_std.plot(x_iters, stds, label="training std", color="tab:red")
        ax_std.legend()
        ax_std.set_title("Final Distance Std")
        ax_std.set_xlabel("Iteration")
        ax_std.set_ylabel("Std [m]")
    ax_std.grid(True, alpha=0.3)

    mean_delta = [float(np.nanmean(delta)) for delta in dist_delta_hist]
    ax_delta.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_delta.plot(mean_delta, color="black", linewidth=2.0, label="mean")
    for env_idx in range(n_envs_max):
        ax_delta.plot([delta[env_idx] for delta in dist_delta_hist], label=f"env {env_idx}", alpha=0.8)
    ax_delta.legend()
    ax_delta.set_title("Distance Change Per Env")
    ax_delta.set_xlabel("Iteration")
    ax_delta.set_ylabel("Delta from Iter 1 [m]")
    ax_delta.grid(True, alpha=0.3)

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
    if save_filepath is None:
        save_filepath = save_dir / "surco_prior.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    if save_filepath2 is not None:
        plt.savefig(save_filepath2, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    plt.close()
    if open_after_save:
        print(f"xdg-open {save_filepath}")
        os.system(f"xdg-open {save_filepath}")


def plot_perturbation_hist(
    c: np.ndarray,        # (n_envs, ACTION_DIM)
    c_pert_ks: list,      # M x (n_envs, ACTION_DIM)
    x_ks: list,           # M x (n_envs, ACTION_DIM)
    x_star: np.ndarray,   # (n_envs, ACTION_DIM)
    n_envs: int,
    iteration: int,
) -> plt.Figure:
    c_np = np.asarray(c)
    c_pert_all = np.stack([np.asarray(ck) for ck in c_pert_ks])   # (M, n_envs, ACTION_DIM)
    x_all = np.stack([np.asarray(xk) for xk in x_ks])             # (M, n_envs, ACTION_DIM)
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
        ax_face.bar(range(NUM_FACES), np.bincount(face_choices_env, minlength=NUM_FACES),
                    color="steelblue", alpha=0.7)
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
    c: np.ndarray,       # (n_envs, ACTION_DIM)
    x_star: np.ndarray,  # (n_envs, ACTION_DIM)
    iteration: int,
) -> plt.Figure:
    """Histogram of raw network outputs across environments — same column layout as plot_perturbation_hist."""
    c_np = np.asarray(c)       # (n_envs, ACTION_DIM)
    x_np = np.asarray(x_star)  # (n_envs, ACTION_DIM)

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

    fig.tight_layout()
    return fig

"""This script trains a SurCo model to solve the PushT problem.

Definitions, single-step problem:
  - y: problem instance / context vector / Dim=9:
        [0:3]:   T_target_pose
        [4:6]:   T_pose
        [7:9]:   T_velocity
        # [10:11]: pusher_xy

  - x: decision variable vector AKA 'action' / Dim=8
        [0:6]: face. One hot encoding of the face.
        [6]:   contact_point in CONTACT_POINT_BOUNDS = (0.2, 0.8)
        [7]:   angle in ANGLE_BOUNDS = (0.2*pi, 0.8*pi)

  - c: surrogate solver parameters / Dim=8
        [0:6]: face logits  (fed to Gurobi; gradient via randomized smoothing)
        [6]:   contact_point target in CONTACT_POINT_BOUNDS
        [7]:   angle target in ANGLE_BOUNDS

  - omega: set of feasible solutions.
        x[0:6] must be one-hot, x[6] in CONTACT_POINT_BOUNDS, x[7] in ANGLE_BOUNDS.
        All constraints enforced by the Gurobi solve.

  - f(x; y): objective function / Dim=1
        The cost is the sum squared distance of the corners of the target block
        (p0, p3, p4, p7) to the target T's corners at the final timestep.


Surco methods:
- SurCo-zero:
    Gradient descent performed directly on c for a single fixed problem instance y.
- SurCo-prior:
    A neural network is trained to predict c conditioned on y, where y is drawn
    from the distribution of problem instances (random resets each iteration).
- SurCo-hybrid:
    Run SurCo-prior to get the NN y → c, then fine-tune c via gradient descent
    on a specific instance y.


System design (SurCo-prior):
1. NN: y → c  (6 face logits + per-face bounded targets for contact_point/angle)
2. Combinatorial solver: Gurobi MIQP
       argmin face_costs^T x_face
            + w_cp (cp - cp_ref(face))^2
            + w_ang (ang - ang_ref(face))^2
       s.t.
       sum(x[0:6])=1, x[0:6]∈{0,1}^6
       x[6] ∈ CONTACT_POINT_BOUNDS  (continuous)
       x[7] ∈ ANGLE_BOUNDS          (continuous)
   Gradient via Berthet et al. 2020 randomized smoothing through the solve.
3. Rollout via step_pure_soft (differentiable physics; one-hot face).
4. Loss: run one rollout using the clean Gurobi solution x* for the predicted
   face logits and continuous targets.
5. Backprop through the rollout, then use randomized smoothing only in the
   VJP to estimate gradients through the combinatorial solve before the NN.


# --- Example usage
unset LD_LIBRARY_PATH  # < note: this runs things with the gpu. Note that it's faster to run on the cpu with nenvs<4


RANDOM_MODE="fixed-spawn__fixed-target" # easy
RANDOM_MODE="random-spawn__fixed-target" # middle
RANDOM_MODE="random-spawn__random-target" # hardest


# Random mode examples
python scripts/main_surco.py --n-envs 1  --verbosity 1 --random-mode ${RANDOM_MODE} --record-video
python scripts/main_surco.py --n-envs 2  --verbosity 1 --random-mode ${RANDOM_MODE} --lr 0.05 --no-wandb --optimizer sgd
python scripts/main_surco.py --n-envs 4  --verbosity 1 --random-mode ${RANDOM_MODE} --lr 0.05 --optimizer sgd
python scripts/main_surco.py --n-envs 32 --verbosity 1 --random-mode ${RANDOM_MODE}
"""

from __future__ import annotations
import json
import os
from time import time

PROGRAM_START_TIME = time()
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from termcolor import cprint
import wandb
import jax

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax_pusht619"))
import jax.numpy as jnp
import numpy as np
import argparse
import optax
import flax.linen as nn
import flax.traverse_util
from pusht619.models import ActionSolver, ActionSolverMultiStep
from pusht619.core import Action, PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES, CONTEXT_DIM_RELATIVE
from pusht619.plotting_utils import plot_results, plot_perturbation_hist, plot_network_output_hist, plot_rollout

_CP_LO, _CP_HI = CONTACT_POINT_BOUNDS
_ANG_LO, _ANG_HI = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
_CP_MID = 0.5 * (_CP_LO + _CP_HI)
_ANG_MID = 0.5 * (_ANG_LO + _ANG_HI)
_CP_SCALE = _CP_HI - _CP_LO
_ANG_SCALE = _ANG_HI - _ANG_LO


class SurCoMLP(nn.Module):
    """Maps context y → solver parameters c (face logits + bounded cp/angle targets).

    Output blocks of size (NUM_FACES + 2):
      [:NUM_FACES]  face logits (unbounded, fed to Gurobi as linear costs)
      [NUM_FACES]   contact_point target, tanh-squashed into CONTACT_POINT_BOUNDS
      [NUM_FACES+1] angle target, tanh-squashed into ANGLE_BOUNDS
    """

    hidden_dims: tuple
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            # x = nn.relu(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.output_dim)(x)
        x = x.reshape(x.shape[0], -1, NUM_FACES + 2)

        # Shift continuous values to the center of their bounds
        x = x.at[:, :, NUM_FACES].add(_CP_MID)
        x = x.at[:, :, NUM_FACES + 1].add(_ANG_MID)
        return x.reshape(x.shape[0], -1)


def save_mlp_weights(filepath: Path, params) -> None:
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    np.savez(filepath, **{k: np.asarray(v) for k, v in flat.items()})


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 10000
DEFAULT_LR = 0.001
N_SIM_STEPS = 25
RESET_SEED = 0
ACTION_DIM = NUM_FACES + 2
M_ROLLOUTS = 9

# Regularization scales
FACE_OUTPUT_REG_BETA = 0.25
CONTACT_POINT_REG_BETA = 0.1
ANGLE_REG_BETA = 0.1

CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0
USE_MC_FOR_CONTINUOUS_GRADIENTS = False


# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 0.5
# PERTURB_LAMBDA = 1.25
CONTINUOUS_PERTURB_SCALE = 0.1

# Evaluation frequency cfg
RANDOM_T_POSE_EVAL_EVERY = 50
BASELINE_EVAL_EVERY = 5

#
_SOLVER = ActionSolver()
_ENV: PushTEnv | None = None
_ENV_BACKWARD: PushTEnv | None = None

_BACKWARD_LOG_DIR: Path | None = None
_CURRENT_ITERATION: int = -1
_LAST_GRAD_X: np.ndarray | None = None
_LAST_BACKWARD_T_GUROBI_MS: float = 0.0
_LAST_BACKWARD_T_PHYSICS_MS: float = 0.0


def _configure_solver(multi_step_n_actions: int | None) -> None:
    global _SOLVER
    if multi_step_n_actions is None:
        _SOLVER = ActionSolver()
        return
    if multi_step_n_actions < 1:
        raise ValueError("multi_step_n_actions must be >= 1")
    _SOLVER = ActionSolverMultiStep(n_actions=multi_step_n_actions)


def _configure_env(env: PushTEnv) -> None:
    global _ENV
    _ENV = env


def _configure_backward_env(env: PushTEnv) -> None:
    global _ENV_BACKWARD
    _ENV_BACKWARD = env


def _configure_backward_log_dir(path: Path) -> None:
    global _BACKWARD_LOG_DIR
    _BACKWARD_LOG_DIR = path


def _run_rollout(
    face_weights: jnp.ndarray,  # (N, n_actions, NUM_FACES) — one-hot
    cp: jnp.ndarray,  # (N, n_actions)
    ang: jnp.ndarray,  # (N, n_actions)
    data,
    env: PushTEnv | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run all action blocks sequentially through step_pure.

    Returns (mean_final_dist scalar, t_distances (N, total_steps), jpos_traj (N, total_steps, dofs)).
    Differentiable w.r.t. cp and ang.
    Pass env explicitly to use a different environment (e.g. _ENV_BACKWARD).
    """
    env = env or _ENV
    assert env is not None, "call _configure_env(env) before training"
    n_actions = face_weights.shape[1]
    rollout_data = data
    t_distances_parts = []
    jpos_traj_parts = []
    for action_idx in range(n_actions):
        rollout_data, _, t_dists, jpos = env.step_pure(
            data=rollout_data,
            face=jnp.argmax(face_weights[:, action_idx, :], axis=-1),
            contact_point=cp[:, action_idx],
            angle=ang[:, action_idx],
            n_sim_steps=N_SIM_STEPS,
            check_t_displacement=False,
        )
        t_distances_parts.append(t_dists)
        jpos_traj_parts.append(jpos)
    t_distances = jnp.concatenate(t_distances_parts, axis=1)
    jpos_traj = jnp.concatenate(jpos_traj_parts, axis=1)
    return jnp.nanmean(t_distances[:, -1]), t_distances, jpos_traj


def _n_actions(action_dim: int) -> int:
    if action_dim % ACTION_DIM != 0:
        raise ValueError(f"Action dimension must be divisible by {ACTION_DIM}, got {action_dim}")
    return action_dim // ACTION_DIM


def _gurobi_solve_batch(c_batch: np.ndarray) -> np.ndarray:
    """Solve the Gurobi objective for every env. Returns (N, action_dim)."""
    return _SOLVER.solve_batch(c_batch)


def _solve_milp_pure_callback(c: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around _gurobi_solve_batch. c: (N,D) → x: (N,D)."""
    c = c.astype(jnp.float32)
    shape = jax.ShapeDtypeStruct(c.shape, jnp.float32)
    return jax.pure_callback(
        lambda cb_c: _gurobi_solve_batch(np.asarray(cb_c)).astype(np.float32),
        shape,
        c,
    )


@jax.custom_vjp
def milp_solver(c: jnp.ndarray, data, rng: jnp.ndarray, verbosity: int) -> jnp.ndarray:
    """Differentiable Gurobi solve: c (N,D) → x_star (N,D).

    Backward: M_ROLLOUTS Monte Carlo draws, each with an actual physics rollout.
        ε_k ~ N(0, I) (face logits only),  x_k = solve(c + λ ε_k)
        L_k = rollout_cost(x_k)            (true physics cost)
        ∂L/∂c_face  ≈ (1/(M λ)) Σ_k ε_k (L_k - mean(L))   [MC estimator, mean-baseline]
        ∂L/∂c_cont  ≈ (1/M) Σ_k ∂L_k/∂c_cont               [averaged analytical grad]
    """
    return _solve_milp_pure_callback(c)


def _milp_forward(c, data, rng, verbosity):
    x_star = _solve_milp_pure_callback(c)
    rng, sample_rng = jax.random.split(rng)
    return x_star, (c, x_star, data, sample_rng, int(verbosity))


def _milp_backward(res, grad_x):
    c, x_star, data, sample_rng, verbosity = res
    n_envs = c.shape[0]
    n_actions = _n_actions(c.shape[1])
    assert _ENV_BACKWARD is not None, "call _configure_backward_env before training"

    # Continuous params are the same for all M perturbed solves (only face logits are perturbed).
    x_star_blocks = x_star.reshape(n_envs, n_actions, ACTION_DIM)
    cp_0 = x_star_blocks[:, :, NUM_FACES]  # (N, n_actions)
    ang_0 = x_star_blocks[:, :, NUM_FACES + 1]  # (N, n_actions)

    key = sample_rng
    face_weights_ks: list = []
    cp_ks_list: list = []
    ang_ks_list: list = []
    eps_faces: list = []
    x_ks: list = []
    c_pert_ks: list = []

    if verbosity > 0:
        jax.debug.print("c=\n{c}", c=c)

    # Gurobi pure_callbacks are inherently sequential — loop only for solves.
    t_gurobi_start = time()
    for k_i in range(M_ROLLOUTS):
        eps_face = jnp.zeros_like(c)
        for action_idx in range(n_actions):
            key, subkey = jax.random.split(key)
            lo = action_idx * ACTION_DIM
            eps_face = eps_face.at[:, lo : lo + NUM_FACES].set(
                jax.random.normal(subkey, (n_envs, NUM_FACES), dtype=jnp.float32)
            )
            if USE_MC_FOR_CONTINUOUS_GRADIENTS:
                key, subkey = jax.random.split(key)
                eps_face = eps_face.at[:, lo + NUM_FACES : lo + NUM_FACES + 2].set(
                    CONTINUOUS_PERTURB_SCALE * jax.random.normal(subkey, (n_envs, 2), dtype=jnp.float32)
                )
        c_pert = (c + PERTURB_LAMBDA * eps_face).astype(jnp.float32)
        x_k = _solve_milp_pure_callback(c_pert)
        x_k_blocks = x_k.reshape(n_envs, n_actions, ACTION_DIM)
        face_weights_ks.append(x_k_blocks[:, :, :NUM_FACES])  # (N, n_actions, F) one-hot
        cp_ks_list.append(x_k_blocks[:, :, NUM_FACES])
        ang_ks_list.append(x_k_blocks[:, :, NUM_FACES + 1])
        eps_faces.append(eps_face)
        x_ks.append(x_k)
        c_pert_ks.append(c_pert)
    t_gurobi = time() - t_gurobi_start

    # Stack all M face-weight arrays → (M*N, n_actions, F); tile data → (M*N, ...).
    # All M rollouts share the same cp/ang (only face differs), so we tile those too
    # and differentiate through the tiling to get the averaged continuous gradient.
    face_weights_all = jnp.concatenate(face_weights_ks, axis=0)  # (M*N, n_actions, F)
    data_tiled = jax.tree.map(lambda x: jnp.repeat(x, M_ROLLOUTS, axis=0), data)

    cp_all = jnp.concatenate(cp_ks_list, axis=0)
    ang_all = jnp.concatenate(ang_ks_list, axis=0)

    def all_rollouts_cost(cp, ang):
        if USE_MC_FOR_CONTINUOUS_GRADIENTS:
            cp_tiled = cp
            ang_tiled = ang
        else:
            # cp: (N, n_actions) — tiled inside so grad flows back to this shape.
            cp_tiled = jnp.repeat(cp, M_ROLLOUTS, axis=0)  # (M*N, n_actions)
            ang_tiled = jnp.repeat(ang, M_ROLLOUTS, axis=0)
        _, t_dists, _ = _run_rollout(face_weights_all, cp_tiled, ang_tiled, data_tiled, _ENV_BACKWARD)
        # t_dists: (M*N, total_steps) → per-rollout costs (M, N)
        costs_per_env = t_dists[:, -1].reshape(M_ROLLOUTS, n_envs)
        mc_rollout_costs_per_perturbation = jnp.nanmean(costs_per_env, axis=1)
        return mc_rollout_costs_per_perturbation.mean(), (mc_rollout_costs_per_perturbation, costs_per_env)

    # One parallel backward pass: mc_rollout_costs_per_perturbation for MC face estimator, grads for continuous.
    t_physics_start = time()
    if USE_MC_FOR_CONTINUOUS_GRADIENTS:
        (_, (mc_rollout_costs_per_perturbation, costs_per_env_all)) = all_rollouts_cost(cp_all, ang_all)
    else:
        (_, (mc_rollout_costs_per_perturbation, costs_per_env_all)), (grad_cp_0, grad_ang_0) = jax.value_and_grad(
            all_rollouts_cost, argnums=(0, 1), has_aux=True
        )(cp_0, ang_0)
    t_physics = time() - t_physics_start
    global _LAST_BACKWARD_T_GUROBI_MS, _LAST_BACKWARD_T_PHYSICS_MS
    _LAST_BACKWARD_T_GUROBI_MS = t_gurobi * 1000
    _LAST_BACKWARD_T_PHYSICS_MS = t_physics * 1000
    # grad_cp_0 = (1/M) Σ_k ∂L_k/∂cp  (chain rule through jnp.repeat averages over M)

    # Face MC gradient — mean-baseline control variate reduces variance.
    mc_rollout_cost_mean = mc_rollout_costs_per_perturbation.mean()
    mean_cost_per_env = jnp.nanmean(costs_per_env_all, axis=0)
    grad_c_face = jnp.zeros_like(c)
    grad_c_mc_cont = jnp.zeros_like(c)

    for k_i in range(M_ROLLOUTS):
        # We divide diff by n_envs because task_loss is a mean over all envs,
        # so the gradient of the task_loss is 1/N * gradient of the env cost.
        diff = (costs_per_env_all[k_i] - mean_cost_per_env) / n_envs
        eps_f = eps_faces[k_i]

        for action_idx in range(n_actions):
            lo = action_idx * ACTION_DIM

            grad_c_face = grad_c_face.at[:, lo : lo + NUM_FACES].add(
                eps_f[:, lo : lo + NUM_FACES] * diff[:, None] / (M_ROLLOUTS * PERTURB_LAMBDA)
            )

            if USE_MC_FOR_CONTINUOUS_GRADIENTS:
                # eps_f for continuous was scaled by CONTINUOUS_PERTURB_SCALE, true_eps = eps_f / CONTINUOUS_PERTURB_SCALE
                # The injected noise std dev = CONTINUOUS_PERTURB_SCALE * PERTURB_LAMBDA
                # grad = true_eps / std * cost = eps_f / (CONTINUOUS_PERTURB_SCALE**2 * PERTURB_LAMBDA) * cost
                grad_c_mc_cont = grad_c_mc_cont.at[:, lo + NUM_FACES : lo + NUM_FACES + 2].add(
                    eps_f[:, lo + NUM_FACES : lo + NUM_FACES + 2]
                    * diff[:, None]
                    / (M_ROLLOUTS * PERTURB_LAMBDA * (CONTINUOUS_PERTURB_SCALE**2))
                )

    if USE_MC_FOR_CONTINUOUS_GRADIENTS:
        grad_c = grad_c_face + grad_c_mc_cont
    else:
        grad_c_continuous = jnp.zeros_like(c)
        for action_idx in range(n_actions):
            lo = action_idx * ACTION_DIM
            grad_c_continuous = grad_c_continuous.at[:, lo + NUM_FACES].set(grad_cp_0[:, action_idx])
            grad_c_continuous = grad_c_continuous.at[:, lo + NUM_FACES + 1].set(grad_ang_0[:, action_idx])
        grad_c = grad_c_face + grad_c_continuous

    if verbosity > 1:
        jax.debug.print("  mc_rollout_cost_mean={mc_rollout_cost_mean}", mc_rollout_cost_mean=mc_rollout_cost_mean)

        def print_env_details(c_val, c_perts_val, x_val, costs_val):
            for env_idx in range(c_val.shape[0]):
                print(f"--- Env {env_idx} ---")
                c_str = np.array2string(c_val[env_idx], precision=4, suppress_small=True)
                print(f"  c: {c_str}")
                for k in range(c_perts_val.shape[0]):
                    faces = []
                    for a in range(c_val.shape[1] // ACTION_DIM):
                        lo = a * ACTION_DIM
                        face = int(np.argmax(x_val[k, env_idx, lo : lo + NUM_FACES]))
                        faces.append(face)
                    c_pert_str = np.array2string(c_perts_val[k, env_idx], precision=4, suppress_small=True)
                    print(f"  Rollout {k}: c_pert={c_pert_str}, face={faces}, cost={costs_val[k, env_idx]:.4f}")

        jax.debug.callback(
            print_env_details, c, jnp.stack(c_pert_ks, axis=0), jnp.stack(x_ks, axis=0), costs_per_env_all
        )
    # cprint(
    #     f"  [backward]  gurobi ({M_ROLLOUTS * n_envs} solves): {t_gurobi * 1000:.0f} ms  |  "
    #     f"physics rollouts + grad: {t_physics * 1000:.0f} ms",
    #     "white",
    # )

    global _LAST_GRAD_X
    _LAST_GRAD_X = np.asarray(grad_x)
    if _BACKWARD_LOG_DIR is not None and _CURRENT_ITERATION >= 0:
        np.savez(
            _BACKWARD_LOG_DIR / f"{_CURRENT_ITERATION:03d}.npz",
            c=np.asarray(c),
            x_star=np.asarray(x_star),
            grad_x=np.asarray(grad_x),
            eps_faces=np.stack([np.asarray(e) for e in eps_faces], axis=0),  # (M, N, D)
            x_perturbed=np.stack([np.asarray(xk) for xk in x_ks], axis=0),  # (M, N, D)
            c_perturbed=np.stack([np.asarray(ck) for ck in c_pert_ks], axis=0),  # (M, N, D)
            costs_per_env=np.asarray(costs_per_env_all),  # (M, N)
            mc_rollout_costs_per_perturbation=np.asarray(mc_rollout_costs_per_perturbation),  # (M,)
            grad_c=np.asarray(grad_c),
            grad_c_face=np.asarray(grad_c_face),
            # grad_c_continuous=np.asarray(grad_c_continuous),
        )

    if wandb.run is not None and _CURRENT_ITERATION >= 0 and (_CURRENT_ITERATION % BASELINE_EVAL_EVERY) == 0:
        fig = plot_perturbation_hist(c, c_pert_ks, x_ks, x_star, n_envs, _CURRENT_ITERATION)
        wandb.log({"milp_backward/c_perturbation_hist": wandb.Image(fig)}, step=_CURRENT_ITERATION)
        plt.close(fig)

    return (grad_c, None, None, None)


milp_solver.defvjp(_milp_forward, _milp_backward)


from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingState:
    iteration_json_path: Path
    iteration: int
    loss: float
    mean_dist_delta: float
    final_dists_np: np.ndarray
    c_batch: np.ndarray
    x_batch: np.ndarray
    grad_c: Optional[np.ndarray] = None
    baseline_means_per_env: Optional[np.ndarray] = None
    pct_vs_baseline_per_env: Optional[np.ndarray] = None
    grad_x: Optional[np.ndarray] = None
    t_poses: Optional[np.ndarray] = None
    t_velocities: Optional[np.ndarray] = None
    target_poses: Optional[np.ndarray] = None

    def save_json(self):
        iteration_payload = {
            "iteration": self.iteration,
            "loss": float(self.loss),
            "mean_dist_delta": float(self.mean_dist_delta),
            "final_distance_per_env": self.final_dists_np.tolist(),
            "c": np.asarray(self.c_batch).tolist(),
            "x": self.x_batch.tolist(),
            "dloss_dc": self.grad_c.tolist() if self.grad_c is not None else None,
            "dloss_dx": None if self.grad_x is None else np.asarray(self.grad_x).tolist(),
            "baseline_mean_per_env": None
            if self.baseline_means_per_env is None
            else np.asarray(self.baseline_means_per_env).tolist(),
            "pct_vs_baseline_per_env": None
            if self.pct_vs_baseline_per_env is None
            else np.asarray(self.pct_vs_baseline_per_env).tolist(),
            "mean_pct_vs_baseline": None
            if self.pct_vs_baseline_per_env is None
            else float(np.nanmean(self.pct_vs_baseline_per_env)),
            "t_poses": None if self.t_poses is None else np.asarray(self.t_poses).tolist(),
            "t_velocities": None if self.t_velocities is None else np.asarray(self.t_velocities).tolist(),
            "target_poses": None if self.target_poses is None else np.asarray(self.target_poses).tolist(),
        }
        self.iteration_json_path.write_text(json.dumps(iteration_payload, indent=2))


def center_action_baseline(
    eval_env: PushTEnv,
    target_poses: np.ndarray,  # (nenvs, 3)
    t_poses: np.ndarray,  # (nenvs, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Try center contact point and angle on each face for all envs.

    Returns (mean_per_env, std_per_env) each shape (nenvs,), where mean/std
    are computed across the NUM_FACES faces.
    """
    nenvs = eval_env.nenvs
    cp_center = np.full((nenvs, 1), _CP_MID, dtype=np.float32)
    ang_center = np.full((nenvs, 1), _ANG_MID, dtype=np.float32)
    face_dists = []
    for face_idx in range(NUM_FACES):
        eval_env.reset(target_poses=target_poses, t_poses=t_poses)
        action = Action(
            face=np.full((nenvs, 1), face_idx, dtype=np.int32),
            contact_point=cp_center,
            angle=ang_center,
        )
        result = eval_env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
        face_dists.append(np.asarray(result.t_distances)[:, -1])  # (nenvs,)
    face_dists_arr = np.stack(face_dists, axis=0)  # (NUM_FACES, nenvs)
    return face_dists_arr.mean(axis=0), face_dists_arr.std(axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    problem_type: str,
    n_envs: int,
    verbosity: int,
    random_mode: str,
    record_video: bool,
    multi_step_n_actions: int | None,
    disable_random: bool,
    use_wandb: bool = True,
    lr: float | None = None,
    perturb_lambda: float | None = None,
    optimizer_type: str = "adam",
    n_opt_steps: int | None = None,
):
    global _CURRENT_ITERATION, PERTURB_LAMBDA, N_OPT_STEPS
    if perturb_lambda is not None:
        PERTURB_LAMBDA = perturb_lambda
    if n_opt_steps is not None:
        N_OPT_STEPS = n_opt_steps
    assert problem_type in ["single_step", "multi_step"], "problem_type must be 'single_step' or 'multi_step'."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    is_multi_step = multi_step_n_actions is not None
    n_actions = multi_step_n_actions if is_multi_step else 1
    _configure_solver(multi_step_n_actions)
    lr = lr if lr is not None else DEFAULT_LR

    env = PushTEnv(
        nenvs=n_envs, record_video=record_video, visualize=False, use_relative_coordinates=True, random_mode=random_mode
    )
    _configure_env(env)
    backward_env = PushTEnv(
        nenvs=n_envs * M_ROLLOUTS,
        record_video=False,
        visualize=False,
        use_relative_coordinates=True,
        random_mode=random_mode,
    )
    _configure_backward_env(backward_env)
    baseline_env = (
        None
        if disable_random
        else PushTEnv(
            nenvs=n_envs, record_video=False, visualize=False, use_relative_coordinates=True, random_mode=random_mode
        )
    )
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    solver_output_dim = ACTION_DIM * n_actions
    random_pose_str = random_mode
    multi_step_str = "multi-step" if is_multi_step else "single-step"
    save_dir = Path(f"logs/{now}__n-envs:{n_envs}__lr:{lr}__{random_pose_str}__{multi_step_str}")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir = save_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)
    backward_dir = save_dir / "backward"
    backward_dir.mkdir(parents=True, exist_ok=True)
    _configure_backward_log_dir(backward_dir)
    os.system(f"xdg-open {save_dir}")

    print("=================")
    print("Saving to: ")
    print(f"save_dir:        {save_dir}")
    print(f"checkpoints_dir: {checkpoints_dir}")
    print(f"iterations_dir:  {iterations_dir}")
    print(f"backward_dir:    {backward_dir}")
    print("=================")

    if use_wandb:
        import wandb

        wandb.init(
            project="pusht619-surco",
            name=save_dir.name,
            config=dict(
                continuous_grad_from_mc=USE_MC_FOR_CONTINUOUS_GRADIENTS,
                n_envs=n_envs,
                lr=lr,
                n_opt_steps=N_OPT_STEPS,
                n_sim_steps=N_SIM_STEPS,
                m_rollouts=M_ROLLOUTS,
                perturb_lambda=PERTURB_LAMBDA,
                face_output_reg_beta=FACE_OUTPUT_REG_BETA,
                contact_point_reg_beta=CONTACT_POINT_REG_BETA,
                angle_reg_beta=ANGLE_REG_BETA,
                baseline_eval_every=BASELINE_EVAL_EVERY,
                num_faces=NUM_FACES,
                problem_type=problem_type,
                n_actions=n_actions,
                random_mode=random_mode,
                relative_coordinates=True,
                checkpoints_dir=checkpoints_dir,
                iterations_dir=iterations_dir,
                backward_dir=backward_dir,
                optimizer=optimizer_type,
            ),
        )

    mlp = SurCoMLP(hidden_dims=(128, 128), output_dim=solver_output_dim)
    params = mlp.init(jax.random.PRNGKey(0), jnp.zeros((1, CONTEXT_DIM_RELATIVE)))

    if optimizer_type == "sgd":
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(lr),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
    opt_state = optimizer.init(params)

    def cost_from_c(c, data, rng_solve, solver_verbosity: int, log_forward: bool):
        if n_envs > 1 and "random-target" in random_mode:

            def _check_c(c_np):
                if np.all(c_np[0] == c_np[1]):
                    print(f"[CHECK FAIL] c is identical across envs: {c_np}")
                    assert False

            jax.debug.callback(_check_c, c)

        # Do one clean forward solve/rollout. The M_ROLLOUTS Monte Carlo rollouts
        # used for gradient estimation live in milp_solver's custom VJP.
        x_star = milp_solver(c, data, rng_solve, solver_verbosity)  # (n_envs, action_dim)
        x_star_blocks = x_star.reshape((x_star.shape[0], n_actions, ACTION_DIM))
        c_blocks = c.reshape((c.shape[0], n_actions, ACTION_DIM))
        face_weights = x_star_blocks[:, :, :NUM_FACES]
        contact_points = x_star_blocks[:, :, NUM_FACES]
        angles = x_star_blocks[:, :, NUM_FACES + 1]
        face_idx = jnp.argmax(face_weights, axis=-1)

        face_weights_in = jax.nn.one_hot(face_idx, NUM_FACES)
        task_loss, t_distances, jpos_traj = _run_rollout(face_weights_in, contact_points, angles, data)

        # Regularize c_blocks (NN outputs) directly, not contact_points/angles
        # from x_star. Those flow through the Gurobi VJP which replaces the
        # continuous gradient with an MC estimate and discards grad_x, so any
        # loss term routed through x_star[cont] gets zero gradient back to c.
        c_cp = c_blocks[:, :, NUM_FACES]
        c_angle = c_blocks[:, :, NUM_FACES + 1]
        face_logit_regularization = FACE_OUTPUT_REG_BETA * jnp.mean(jnp.square(c_blocks[:, :, :NUM_FACES]))
        cp_regularization = CONTACT_POINT_REG_BETA * jnp.mean(jnp.square((c_cp - _CP_MID) / _CP_SCALE))
        angle_regularization = ANGLE_REG_BETA * jnp.mean(jnp.square((c_angle - _ANG_MID) / _ANG_SCALE))
        loss = task_loss + face_logit_regularization + cp_regularization + angle_regularization

        if log_forward and verbosity > 0:
            jax.debug.print(
                "sum(is_nan)={n} task_loss={task_loss:.6f} face_logit_regularization={flr:.6f} cp_regularization={cpr:.6f} angle_regularization={ar:.6f}",
                n=jnp.sum(jnp.isnan(t_distances[:, -1])),
                task_loss=task_loss,
                flr=face_logit_regularization,
                cpr=cp_regularization,
                ar=angle_regularization,
            )
        if log_forward and verbosity > 1:
            jax.debug.print("final_dists=\n{d}", d=t_distances[:, -1])
        return loss, (
            t_distances,
            jpos_traj,
            face_weights,
            contact_points,
            angles,
            c,
            task_loss,
            face_logit_regularization,
            cp_regularization,
            angle_regularization,
        )

    def cost(params, data, rng_solve):
        ctx = env.get_context_vector(data)  # (n_envs, 9)

        if n_envs > 1:

            def _check_ctx_all(ctx_np):
                dupes = [i for i in range(1, n_envs) if np.allclose(ctx_np[i], ctx_np[0], atol=1e-6)]
                if dupes and "random" in random_mode:
                    cprint(f"[CHECK FAIL] ctx identical to env0 for envs {dupes}", "red")
                    for i in [0] + dupes:
                        cprint(f"  ctx[{i}]={ctx_np[i]}", "red")

            jax.debug.callback(_check_ctx_all, ctx)

        c = mlp.apply(params, ctx)  # (n_envs, action_dim)

        if n_envs > 1:

            def _check_c_all(c_np):
                dupes = [i for i in range(1, n_envs) if np.allclose(c_np[i], c_np[0], atol=1e-6)]
                if dupes and "random" in random_mode:
                    cprint(f"[CHECK FAIL] c identical to env0 for envs {dupes}", "red")

            jax.debug.callback(_check_c_all, c)

        if verbosity > 0:
            jax.debug.print("context=\n{ctx}", ctx=ctx)
        return cost_from_c(c, data, rng_solve, verbosity, True)

    # Do NOT wrap in jax.jit: pure_callback dispatches to Python per Gurobi
    # solve, so a JIT wrapper adds overhead without benefit. Physics is already
    # JIT'd internally.
    # Training step: computes loss + gradient of loss w.r.t. MLP params.
    # The grad flows: loss → x_star → c → params (via Gurobi VJP + autograd through MLP).
    loss_and_grad_wrt_params = jax.value_and_grad(cost, argnums=0, has_aux=True)

    # Logging only: computes gradient of loss w.r.t. c (the NN output / Gurobi input).
    # This is NOT used for training — it lets us inspect what signal is arriving at c
    # before it flows back through the MLP, useful for debugging the Gurobi VJP.
    # grad_wrt_c = jax.grad(lambda c, data, rng_solve: cost_from_c(c, data, rng_solve, 0, False)[0], argnums=0)

    print("SurCo-prior: training NN  y → solver params  (Gurobi + randomized-smoothing VJP)")
    dist_change_hist = []  # list of (n_envs,) float — change in distance from start of rollout per env
    face_hist = []  # list of (n_envs,) int — argmax face per env per iter
    cp_hist = []  # list of (n_envs,) float — contact_point per env per iter
    ang_hist = []  # list of (n_envs,) float — angle per env per iter
    c_face_hist = []  # list of (n_envs, NUM_FACES) float — face logits per env per iter

    lowest_mean_dist_delta = float("inf")
    t_start = time()

    for it in range(N_OPT_STEPS):
        _CURRENT_ITERATION = it
        if it == 0:
            print(f"Program loading time: {time() - PROGRAM_START_TIME:.2f} s")

        print()
        print()
        print(f"|  ────────────────────────────────────────────────────────────────────────────────────────────  |")
        print(
            f"|     ───────────────────────────────────     iter {it + 1:2d}     ──────────────────────────────────     |"
        )
        print()

        is_eval_step = (it % RANDOM_T_POSE_EVAL_EVERY) == 0
        if "random" in random_mode:
            if is_eval_step:
                env.reset(0)
            else:
                env.reset()
        else:
            env.reset(seed=RESET_SEED)

        t0 = time()

        env_data_0 = env.data

        if n_envs > 1:
            jp = np.asarray(env_data_0.joint_positions)
            jp_dupes = [i for i in range(1, n_envs) if np.all(jp[i] == jp[0])]
            if jp_dupes and "random" in random_mode:
                cprint(f"[CHECK FAIL] joint_positions identical to env0 for envs: {jp_dupes}", "red")
                assert False
            t_poses_np = env.t_poses
            tgt_poses_np = env.target_poses
            t_dupes = [i for i in range(1, n_envs) if np.all(t_poses_np[i] == t_poses_np[0])]
            tgt_dupes = [i for i in range(1, n_envs) if np.all(tgt_poses_np[i] == tgt_poses_np[0])]
            if t_dupes and "random─spawn" in random_mode:
                cprint(f"[CHECK FAIL] t_poses identical to env0 for envs: {t_dupes}  values: {t_poses_np}", "red")
                assert False
            if tgt_dupes and "random-target" in random_mode:
                cprint(
                    f"[CHECK FAIL] target_poses identical to env0 for envs: {tgt_dupes}  values: {tgt_poses_np}", "red"
                )
                assert False
            if np.all(tgt_poses_np[0] == tgt_poses_np[1]) and "random-target" in random_mode:
                cprint(f"[CHECK FAIL] target_poses identical: {tgt_poses_np}", "red")
                assert False
            xy = env._xy_centers
            if np.all(xy[0] == xy[1]):
                cprint(f"[CHECK FAIL] _xy_centers identical: {xy}", "red")
                assert False

        step_key = jax.random.PRNGKey(it)
        (
            (
                loss,
                (
                    t_distances,
                    jpos_traj,
                    face_weights,
                    cp_batch,
                    ang_batch,
                    c_batch,
                    task_loss,
                    face_logit_regularization,
                    cp_regularization,
                    angle_regularization,
                ),
            ),
            g_raw,
        ) = loss_and_grad_wrt_params(params, env_data_0, step_key)
        t_forward_backward = time() - t0

        # t1 = time()
        # grad_c = np.asarray(grad_wrt_c(c_batch, env_data_0, step_key))
        # t_grad_c = time() - t1

        t2 = time()
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        updates, opt_state = optimizer.update(g_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        t_optimizer = time() - t2

        final_dists_np = np.asarray(t_distances[:, -1])
        initial_dists_np = np.asarray(t_distances[:, 0])
        dist_deltas_np = final_dists_np - initial_dists_np
        mean_dist_delta = float(np.mean(dist_deltas_np))
        face_idx_np = np.asarray(jnp.argmax(face_weights, axis=-1))
        face_hist_current = face_idx_np[:, 0] if is_multi_step else face_idx_np
        cp_hist_current = np.asarray(cp_batch[:, 0] if is_multi_step else cp_batch)
        ang_hist_current = np.asarray(ang_batch[:, 0] if is_multi_step else ang_batch)
        dt = time() - t0
        x_batch = np.concatenate(
            [
                np.asarray(face_weights),
                np.asarray(cp_batch)[..., None],
                np.asarray(ang_batch)[..., None],
            ],
            axis=-1,
        ).reshape(n_envs, -1)

        # Print NaN environments / gradients
        #
        nan_envs = np.where(np.isnan(final_dists_np))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                cprint(f"  Env: {env_idx} - T distance is NaN", "red")
        n_bad_grads = sum(not jnp.all(jnp.isfinite(x)).item() for x in jax.tree_util.tree_leaves(g_raw))
        if n_bad_grads > 0:
            cprint(
                f"WARNING: {n_bad_grads} non-finite values in raw gradients (sanitized to 0 for this step).",
                "red",
            )

        # Print results
        #
        print(f"faces=          {face_idx_np[:, 0]}")
        print(f"angles=         {ang_hist_current[:, 0]}")
        print(f"contact-points= {cp_hist_current[:, 0]}")
        baseline_means_np = None
        pct_vs_baseline = None
        mean_pct_vs_baseline = None
        t_baseline = 0.0
        is_baseline_step = (it % BASELINE_EVAL_EVERY) == 0
        if not disable_random and is_baseline_step:
            assert baseline_env is not None
            t3 = time()
            baseline_means_np, _ = center_action_baseline(
                baseline_env,
                target_poses=env.target_poses,
                t_poses=env.t_poses,
            )
            t_baseline = time() - t3
            pct_vs_baseline = 100.0 * (final_dists_np - baseline_means_np) / np.maximum(baseline_means_np, 1e-8)
            mean_pct_vs_baseline = float(np.nanmean(pct_vs_baseline))
            print(f"|____ baseline mean per env:  {np.round(baseline_means_np, 5).tolist()}")
            print(f"|____ % vs baseline per env:  {np.round(pct_vs_baseline, 3).tolist()}")
            print(f"|____ mean % vs baseline: {mean_pct_vs_baseline:.4f}%")

        if verbosity > 0:
            cprint(
                f"|____ mean_dist_change: {mean_dist_delta * 100:.3f} [cm]", "green" if mean_dist_delta < 0 else "red"
            )
            print(f"|____ dist_deltas_np=\n{dist_deltas_np}")
        cprint(
            f"|____ timing  fwd+bwd: {t_forward_backward * 1000:.0f} ms  "
            # f"grad_c: {t_grad_c*1000:.0f} ms  "
            f"optimizer: {t_optimizer * 1000:.0f} ms  "
            f"baseline: {t_baseline * 1000:.0f} ms  "
            f"total: {dt * 1000:.0f} ms",
            "white",
        )
        state = TrainingState(
            iteration_json_path=iterations_dir / f"{it:03d}.json",
            iteration=it,
            loss=float(loss),
            mean_dist_delta=mean_dist_delta,
            final_dists_np=final_dists_np,
            c_batch=c_batch,
            x_batch=x_batch,
            grad_c=None,
            baseline_means_per_env=baseline_means_np,
            pct_vs_baseline_per_env=pct_vs_baseline,
            grad_x=_LAST_GRAD_X,
            t_poses=env.t_poses,
            t_velocities=env.t_velocities,
            target_poses=env.target_poses,
        )
        state.save_json()

        # Print gradient statistics
        #
        max_grad, mean_grad = None, None
        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for g in jax.tree_util.tree_leaves(g_params)]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_grad = float(jnp.mean(jnp.array([jnp.mean(g).item() for g in grad_abs_values])))
            print()
            print(f"|____ max |grad|: {max_grad:.6f}, mean |grad|: {mean_grad:.6f}")

        if use_wandb:
            wandb_payload: dict = {
                "loss": float(loss),
                "loss/total": float(loss),
                "loss/task": float(task_loss),
                "loss/face_logit_regularization": float(face_logit_regularization),
                "loss/cp_regularization": float(cp_regularization),
                "loss/angle_regularization": float(angle_regularization),
                "mean_dist_change": mean_dist_delta,
                "mean_final_dist": float(np.mean(final_dists_np)),
                "std_final_dist": float(np.std(final_dists_np)),
                "n_nan_envs": len(nan_envs),
                "n_bad_grads": n_bad_grads,
                "time/iterations_per_second": 1.0 / dt,
                "time/time_per_iteration_ms": dt * 1000,
                "time/forward_backward_ms": t_forward_backward * 1000,
                "time/backward_gurobi_ms": _LAST_BACKWARD_T_GUROBI_MS,
                "time/backward_physics_ms": _LAST_BACKWARD_T_PHYSICS_MS,
                # "time/grad_c_ms": t_grad_c * 1000,
                "time/optimizer_ms": t_optimizer * 1000,
                "time/baseline_ms": t_baseline * 1000,
                # "n_envs_better": int(n_envs_better),
            }
            if mean_pct_vs_baseline is not None and pct_vs_baseline is not None:
                wandb_payload["mean_pct_vs_baseline"] = mean_pct_vs_baseline
                wandb_payload["std_pct_vs_baseline"] = float(np.nanstd(pct_vs_baseline))
            if max_grad is not None:
                wandb_payload["max_grad"] = max_grad
                wandb_payload["mean_grad"] = mean_grad
            wandb.log(wandb_payload, step=it)
            action_log = {}
            for env_i in range(min(n_envs, 4)):
                face_i = int(face_hist_current[env_i]) if is_multi_step else int(face_idx_np[env_i])
                action_log[f"action/face_env_{env_i}"] = face_i
                action_log[f"action/contact_point_env_{env_i}"] = float(cp_hist_current[env_i])
                action_log[f"action/angle_env_{env_i}"] = float(ang_hist_current[env_i])
            wandb.log(action_log, step=it)
            c0 = np.asarray(c_batch)[0]
            # dc0 = np.asarray(grad_c)[0]
            c_log = {f"c/face_{fi}": float(c0[fi]) for fi in range(NUM_FACES)}
            c_log["c/contact_point"] = float(c0[NUM_FACES])
            c_log["c/angle"] = float(c0[NUM_FACES + 1])
            wandb.log(c_log, step=it)
            # dc_log = {f"dc/face_{fi}": float(dc0[fi]) for fi in range(NUM_FACES)}
            # dc_log["dc/contact_point"] = float(dc0[NUM_FACES])
            # dc_log["dc/angle"] = float(dc0[NUM_FACES + 1])
            # wandb.log({**c_log, **dc_log}, step=it)

        # Log results for plotting
        #
        dist_change_hist.append(final_dists_np - initial_dists_np)
        face_hist.append(face_hist_current)
        cp_hist.append(cp_hist_current)
        ang_hist.append(ang_hist_current)
        c_face_hist.append(np.asarray(c_batch)[:, :NUM_FACES])  # (n_envs, NUM_FACES)

        # Save weights and snapshot plot
        #
        if (it + 1) % 5 == 0:
            filepath = checkpoints_dir / f"mlp_iter_{it + 1:03d}.npz"
            save_mlp_weights(filepath, params)
            cprint(f"|____ saved weights to {filepath}", "yellow")
            plot_results(
                dist_change_hist=dist_change_hist,
                face_hist=face_hist,
                cp_hist=cp_hist,
                ang_hist=ang_hist,
                n_envs=n_envs,
                n_opt_steps=N_OPT_STEPS,
                random_mode=random_mode,
                m_rollouts=M_ROLLOUTS,
                perturb_lambda=PERTURB_LAMBDA,
                relative_coordinates=True,
                save_filepath=save_dir / f"{it + 1:03d}.png",
                save_filepath2=save_dir / f"latest.png",
                open_after_save=False,
            )
            t_poses_initial = np.asarray(jpos_traj[:, 0, [env._T_x_idx, env._T_y_idx, env._T_theta_idx]])
            t_poses_final = np.asarray(jpos_traj[:, -1, [env._T_x_idx, env._T_y_idx, env._T_theta_idx]])
            pusher_initial_positions = np.asarray(jpos_traj[:, 0, [env._pusher_x_idx, env._pusher_y_idx]])
            rollout_save_filepath = save_dir / f"rollout_{it + 1:03d}.png"
            plot_rollout(
                save_dir=save_dir,
                t_poses_initial=t_poses_initial,
                t_poses_final=t_poses_final,
                target_poses=env.target_poses,
                pusher_initial_positions=pusher_initial_positions,
                save_filepath=rollout_save_filepath,
            )
            if use_wandb:
                fig_out = plot_network_output_hist(np.asarray(c_batch), np.asarray(x_batch), it)
                wandb.log({"plot": wandb.Image(str(save_dir / "latest.png"))}, step=it)
                wandb.log({"c_figures/network_output_hist": wandb.Image(fig_out)}, step=it)
                wandb.log({"rollout": wandb.Image(str(rollout_save_filepath))}, step=it)
                plt.close(fig_out)
        if mean_dist_delta < lowest_mean_dist_delta and it > 0:
            lowest_mean_dist_delta = mean_dist_delta
            cprint(f"New lowest mean dist: {lowest_mean_dist_delta:.5f} [m]", "green")
            filepath = checkpoints_dir / f"mlp_lowest_mean_dist_delta.npz"
            save_mlp_weights(filepath, params)
            if record_video:
                save_filepath = save_dir / f"best.mp4"
                env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))

        # Save video
        #
        if record_video and it % 5 == 0:
            save_filepath = save_dir / f"{it + 1:03d}.mp4"
            env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
            print(f"  Saved video to {save_filepath}")
            if use_wandb:
                wandb.log({"video": wandb.Video(str(save_filepath))}, step=it)

        if it == 0:
            cprint(f"First iteration time: {time() - t_start:.2f} s", "yellow")

    cprint(f"\nOptimization took {time() - t_start:.2f} s total", "green")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, default="single_step", choices=["single_step", "multi_step"])
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument(
        "--random-mode",
        type=str,
        default="random-spawn__fixed-target",
        choices=[
            "fixed-spawn__fixed-target",
            "random-spawn__fixed-target",
            "random-spawn__random-target",
        ],
        help="Randomization mode for T block spawn and target poses",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--multi-step-n-actions", type=int)
    parser.add_argument("--disable-random", action="store_true", help="Skip random action baseline sampling")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--perturb-lambda", type=float, help="Randomized smoothing scale (PERTURB_LAMBDA)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--n-opt-steps", type=int, default=1000, help="Number of optimization steps")
    args = parser.parse_args()
    assert args.verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    assert args.n_envs is not None, "n_envs must be specified"
    main(
        problem_type=args.problem_type,
        n_envs=args.n_envs,
        verbosity=args.verbosity,
        random_mode=args.random_mode,
        record_video=args.record_video,
        multi_step_n_actions=args.multi_step_n_actions,
        disable_random=args.disable_random,
        use_wandb=not args.no_wandb,
        lr=args.lr,
        perturb_lambda=args.perturb_lambda,
        optimizer_type=args.optimizer,
        n_opt_steps=args.n_opt_steps,
    )

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import gurobipy as gp
from gurobipy import GRB

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 1000
LR = 0.1
N_SIM_STEPS = 50
RESET_SEED = 0
RANDOMZED_SMOOTHING_K = 10
OUTPUT_REG_BETA = 1e-3
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 0.1


# ── Gurobi action solver ──────────────────────────────────────────────────────

_lo_cp, _hi_cp = CONTACT_POINT_BOUNDS
_lo_ang, _hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
_mid_cp = 0.5 * (_lo_cp + _hi_cp)
_mid_ang = 0.5 * (_lo_ang + _hi_ang)


class ActionSolver:
    """Gurobi solver matching the training-time objective."""

    def __init__(self):
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        self.model.setParam("Presolve", 0)
        self.xf = self.model.addVars(NUM_FACES, vtype=GRB.BINARY, name="xf")
        self.cp = self.model.addVar(lb=_lo_cp, ub=_hi_cp, vtype=GRB.CONTINUOUS, name="cp")
        self.ang = self.model.addVar(lb=_lo_ang, ub=_hi_ang, vtype=GRB.CONTINUOUS, name="ang")
        self.model.addConstr(gp.quicksum(self.xf[i] for i in range(NUM_FACES)) == 1)
        self.model.update()

    def solve(self, c: np.ndarray) -> np.ndarray:
        """Update objective from c (8,), solve, return x (8,)."""
        if not np.all(np.isfinite(c)):
            c = np.zeros(NUM_FACES + 2, dtype=np.float32)
            c[NUM_FACES] = _mid_cp
            c[NUM_FACES + 1] = _mid_ang
        cp_ref = float(np.clip(c[NUM_FACES], _lo_cp, _hi_cp))
        ang_ref = float(np.clip(c[NUM_FACES + 1], _lo_ang, _hi_ang))
        face_obj = gp.quicksum(float(c[i]) * self.xf[i] for i in range(NUM_FACES))
        cp_obj = CP_TARGET_WEIGHT * (self.cp - cp_ref) * (self.cp - cp_ref)
        ang_obj = ANG_TARGET_WEIGHT * (self.ang - ang_ref) * (self.ang - ang_ref)
        self.model.setObjective(face_obj + cp_obj + ang_obj, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        face_vals = np.array([self.xf[i].X for i in range(NUM_FACES)], dtype=np.float32)
        x = np.append(face_vals, [self.cp.X, self.ang.X]).astype(np.float32)
        assert x.shape == (NUM_FACES + 2,), f"x must be ({NUM_FACES + 2},), got {x.shape}"
        return x

    def solve_batch(self, c_batch: np.ndarray) -> np.ndarray:
        """Solve the Gurobi objective for every env. Returns (N, {NUM_FACES + 2})."""
        out = np.zeros((c_batch.shape[0], NUM_FACES + 2), dtype=np.float32)
        for i in range(c_batch.shape[0]):
            out[i] = self.solve(c_batch[i])
        return out


class ActionSolverMultiStep:
    """MIQP over a fixed horizon of M consecutive actions.

    Each action m = 0 … M-1 has the same structure as :class:`ActionSolver`, but
    the six face variables are *independent* binaries (no one-hot / no sum
    constraint). The total objective is the sum of the M per-step objectives.

    - ``c`` layout: ``(8 * M,)`` with blocks ``c[8*m : 8*(m+1)]`` =
      ``[face linear costs (6), cp target, ang target]`` for step ``m``.

    - ``x`` layout: ``(8 * M,)`` with blocks ``[face bits (6), cp, ang]`` per step.
    """

    def __init__(self, n_actions: int = 3):
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        self.n_actions = int(n_actions)
        self.action_dim = NUM_FACES + 2
        self.c_dim = self.action_dim * self.n_actions

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        self.model.setParam("Presolve", 0)

        self.xf: list = []
        self.cp: list = []
        self.ang: list = []
        for m in range(self.n_actions):
            self.xf.append(self.model.addVars(NUM_FACES, vtype=GRB.BINARY, name=f"xf_{m}"))
            self.cp.append(self.model.addVar(lb=_lo_cp, ub=_hi_cp, vtype=GRB.CONTINUOUS, name=f"cp_{m}"))
            self.ang.append(self.model.addVar(lb=_lo_ang, ub=_hi_ang, vtype=GRB.CONTINUOUS, name=f"ang_{m}"))
        self.model.update()

    def solve(self, c: np.ndarray) -> np.ndarray:
        """Update objective from c ((NUM_FACES + 2) * n_actions,), solve, return x ((NUM_FACES + 2) * n_actions,)."""
        c = np.asarray(c, dtype=np.float64).ravel()
        if c.shape != (self.c_dim,):
            raise ValueError(f"c must have shape ({self.c_dim},), got {c.shape}")
        if not np.all(np.isfinite(c)):
            c = np.zeros(self.c_dim, dtype=np.float32)
            for m in range(self.n_actions):
                c[(NUM_FACES + 2) * m + NUM_FACES] = _mid_cp
                c[(NUM_FACES + 2) * m + NUM_FACES + 1] = _mid_ang

        terms = []
        for m in range(self.n_actions):
            lo = (NUM_FACES + 2) * m
            cm = c[lo : lo + (NUM_FACES + 2)]
            cp_ref = float(np.clip(cm[NUM_FACES], _lo_cp, _hi_cp))
            ang_ref = float(np.clip(cm[NUM_FACES + 1], _lo_ang, _hi_ang))
            face_obj = gp.quicksum(float(cm[i]) * self.xf[m][i] for i in range(NUM_FACES))
            cp_obj = CP_TARGET_WEIGHT * (self.cp[m] - cp_ref) * (self.cp[m] - cp_ref)
            ang_obj = ANG_TARGET_WEIGHT * (self.ang[m] - ang_ref) * (self.ang[m] - ang_ref)
            terms.append(face_obj + cp_obj + ang_obj)
        obj = terms[0]
        for t in terms[1:]:
            obj = obj + t
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()

        parts: list[np.ndarray] = []
        for m in range(self.n_actions):
            face_vals = np.array([self.xf[m][i].X for i in range(NUM_FACES)], dtype=np.float32)
            parts.append(np.append(face_vals, [self.cp[m].X, self.ang[m].X]).astype(np.float32))
        x = np.concatenate(parts, axis=0)
        assert x.shape == (self.c_dim,), f"x must be ({self.c_dim},), got {x.shape}"
        return x

    def solve_batch(self, c_batch: np.ndarray) -> np.ndarray:
        """Solve the Gurobi objective for every env. Returns (N, (NUM_FACES + 2) * self.n_actions)."""
        c_batch = np.asarray(c_batch)
        if c_batch.ndim != 2 or c_batch.shape[1] != self.c_dim:
            raise ValueError(f"c_batch must be (N, (NUM_FACES + 2) * n_actions), got {c_batch.shape}")
        n = c_batch.shape[0]
        out = np.zeros((n, (NUM_FACES + 2) * self.n_actions), dtype=np.float32)
        for i in range(n):
            out[i] = self.solve(c_batch[i])
        return out

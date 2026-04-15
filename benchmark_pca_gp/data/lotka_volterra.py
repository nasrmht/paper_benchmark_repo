"""Lotka-Volterra dataset with 4 constrained output fields.

System:
    dX/dt =  a*X - b*X*Y
    dY/dt = -c*Y + d*X*Y

Conserved quantity (Hamiltonian, constant in time):
    H(X, Y; b, d) = d*X - c*ln(X) + b*Y - a*ln(Y)

Output fields (per trajectory):
    p = X              (N × S)
    q = Y              (N × S)
    r = ln(Y)          (N × S)
    s = ln(X) + H/c    (N × S)   ← absorbs H into the field definition

Constraint (input-dependent α, HOMOGENEOUS RHS = 0):
    d·p + b·q − a·r − c·s = 0
    with  α^p = d,  α^q = b,  α^r = −a,  α^s = −c.

Derivation of s:
    Original: d·p + b·q − a·r − c·ln(X) = H
    → c·s = c·ln(X) + H  ⟹  s = ln(X) + H/c
    → d·p + b·q − a·r − c·s = H − H = 0  ✓

Because H = 0, the centering pipeline only needs the α-multiplication step
(Case 3 with zero RHS).  constraint_rhs_per_sample() is not overridden
(returns None by default).

GP inputs (varied parameters): b, d   (2-dimensional)
Fixed parameters: a=1.1, c=0.4, x0=(1.9, 0.3)
"""
import numpy as np
from typing import List, Optional, Tuple
from scipy.stats.qmc import LatinHypercube, scale

from .base import Dataset


# ---------------------------------------------------------------------------
# ODE numerics
# ---------------------------------------------------------------------------

def _lv_rhs(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, c, d = params
    dX = a * x[0] - b * x[0] * x[1]
    dY = -c * x[1] + d * x[0] * x[1]
    return np.array([dX, dY])


def _rk4(params: np.ndarray, x0: np.ndarray, t0: float, tf: float, dt: float):
    t = np.arange(t0, tf, dt)
    n = len(t)
    x = np.zeros((2, n))
    x[:, 0] = x0
    for i in range(n - 1):
        k1 = _lv_rhs(x[:, i], params)
        k2 = _lv_rhs(x[:, i] + 0.5 * dt * k1, params)
        k3 = _lv_rhs(x[:, i] + 0.5 * dt * k2, params)
        k4 = _lv_rhs(x[:, i] + dt * k3, params)
        x[:, i + 1] = x[:, i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LotkaVolterraDataset(Dataset):
    """Lotka-Volterra benchmark: 4 output fields under input-dependent constraint.

    Outputs: p=X, q=Y, r=ln(Y), s=ln(X)+H/c.
    Constraint: d·p + b·q − a·r − c·s = 0  (homogeneous, H absorbed into s).

    The Case 3 centering pipeline in Dataset.center() handles the input-dependent
    α = [d, b, -a, -c] with zero RHS:
        z^k_i = α^k_i · y^k_i
    yielding z^k fields that satisfy Σ_j z^j = 0 (unit constraint).

    Parameters
    ----------
    t_end        : end time for integration
    dt           : time step
    b_range      : (b_min, b_max) uniform distribution for parameter b
    d_range      : (d_min, d_max) uniform distribution for parameter d
    n_time_steps : if not None, truncate trajectories to this length
    """

    def __init__(
        self,
        t_end: float = 20.0,
        dt: float = 0.05,
        b_range: Tuple[float, float] = (0.37, 0.40),
        d_range: Tuple[float, float] = (0.00, 0.06),
        n_time_steps: int = None,
    ):
        self.t_end = t_end
        self.dt = dt
        self.b_range = b_range
        self.d_range = d_range
        self.n_time_steps = n_time_steps

        # Fixed parameters
        self._a = 1.1
        self._c = 0.4
        self._x0 = np.array([1.9, 0.3])

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    def constraint_vector(self) -> np.ndarray:
        """Unit constraint [1,1,1,1] satisfied by the z-fields after centering."""
        return np.ones(4)

    @property
    def n_outputs(self) -> int:
        return 4

    @property
    def input_dim(self) -> int:
        return 2   # (b, d)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate(
        self, n_total: int, seed: int = 42
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate n_total samples.

        Returns
        -------
        X_gp   : (N, 2)  GP inputs = [b, d]
        fields : [p, q, r, s]  each (N, S)
            p = X            (prey trajectory)
            q = Y            (predator trajectory)
            r = ln(Y)
            s = ln(X) + H/c  (H = d·X₀ + b·Y₀ − a·ln(Y₀) − c·ln(X₀))

        These fields satisfy d·p + b·q − a·r − c·s = 0 exactly.
        """
        # sampler = LatinHypercube(d=2, seed=seed)
        # lhs = scale(sampler.random(n_total),
        #             l_bounds=[self.b_range[0], self.d_range[0]],
        #             u_bounds=[self.b_range[1], self.d_range[1]])
        # b_vals = lhs[:, 0]
        # d_vals = lhs[:, 1]
        rng = np.random.RandomState(seed)
        b_vals = rng.uniform(*self.b_range, n_total)
        d_vals = rng.uniform(*self.d_range, n_total)

        params_all = np.column_stack([
            np.full(n_total, self._a),
            b_vals,
            np.full(n_total, self._c),
            d_vals,
        ])

        X_list, Y_list = [], []
        for params in params_all:
            traj, _ = _rk4(params, self._x0.copy(), 0.0, self.t_end, self.dt)
            X_t = traj[0]
            Y_t = traj[1]
            if self.n_time_steps is not None:
                X_t = X_t[:self.n_time_steps]
                Y_t = Y_t[:self.n_time_steps]
            X_list.append(X_t)
            Y_list.append(Y_t)

        X_traj = np.array(X_list)   # (N, S)
        Y_traj = np.array(Y_list)   # (N, S)

        # Hamiltonian H(b, d) — scalar per sample, constant along trajectory
        X0, Y0 = self._x0[0], self._x0[1]
        H = (d_vals * X0 + b_vals * Y0
             - self._a * np.log(Y0)
             - self._c * np.log(X0))             # (N,)

        p = X_traj                               # prey
        q = Y_traj                               # predator
        r = np.log(Y_traj)+ (H / self._a)[:, np.newaxis]                       # ln(Y)
        s = np.log(X_traj)    # ln(X) + H/c

        X_gp = np.column_stack([b_vals, d_vals])
        return X_gp, [p, q, r, s]

    # ------------------------------------------------------------------
    # Generalised constraint interface (Case 3, zero RHS)
    # ------------------------------------------------------------------

    def input_weights(self, X: np.ndarray) -> np.ndarray:
        """Per-sample constraint coefficients α(x_i) = [d_i, b_i, −a, −c].

        The constraint is  d·p + b·q − a·r − c·s = 0.
        With  α^p = d,  α^q = b,  α^r = −a,  α^s = −c,
        this is  Σ_j α^j y^j = 0  (Case 3, homogeneous).

        Parameters
        ----------
        X : (N, 2)  columns = [b, d]

        Returns
        -------
        (N, 4) array of α^j(x_i)
        """
        b = X[:, 0]
        d = X[:, 1]
        N = len(X)
        return np.column_stack([
            d,
            b,
            -self._a * np.ones(N),
            -self._c * np.ones(N),
        ])
    # constraint_rhs_per_sample() not overridden → returns None (H = 0)

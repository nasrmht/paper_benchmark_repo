"""ModelRegistry: factory functions for all benchmark model combinations."""
import numpy as np
from typing import List, Optional, Dict, Any

from ..reduction.colwise import ColwisePCA
from ..reduction.fieldwise import FieldwisePCA
from ..regression.sogp_mode import SOGPModeRegressor
from ..regression.mogp_mode import MOGPLCMModeRegressor
from .rowwise_constrained import RowwiseConstrainedModel
from .fixed_output import FixedOutputModel
from .fieldwise_optimized import FieldwiseOptimizedModel


class ModelRegistry:
    """Factory for creating benchmark surrogate models.

    All factory methods accept a ``gp_config`` dict with optional keys:
        n_restarts, maxiter, noise_var, n_kernels, rank
    """

    # ------------------------------------------------------------------
    # Individual model factories
    # ------------------------------------------------------------------

    @staticmethod
    def create_rowwise_constrained(
        n_modes: int,
        u: np.ndarray,
        gp_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> RowwiseConstrainedModel:
        """RC: RowwisePCA + per-mode Constrained MOGP."""
        cfg = gp_config or {}
        return RowwiseConstrainedModel(
            name=name or f"RC_ConstMOGP_M{n_modes}",
            n_modes=n_modes,
            u=u,
            n_kernels=cfg.get("n_kernels", 2),
            rank=cfg.get("rank", None),
            n_restarts=cfg.get("n_restarts", 3),
            maxiter=cfg.get("maxiter", 100),
            noise_var=cfg.get("noise_var", 1e-3),
            seed=cfg.get("seed", None),
        )

    @staticmethod
    def create_colwise_independent(
        n_modes: int,
        u: np.ndarray,
        fixed_idx: int,
        gp_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> FixedOutputModel:
        """CI: ColwisePCA + M independent SOGPs (one per mode)."""
        cfg = gp_config or {}

        def mode_factory():
            return SOGPModeRegressor(
                n_restarts=cfg.get("n_restarts", 3),
                maxiter=cfg.get("maxiter", 100),
                var_noise=cfg.get("noise_var", 1e-3),
                seed=cfg.get("seed", None),
            )

        reducer = ColwisePCA(n_modes=n_modes, fixed_idx=fixed_idx)
        return FixedOutputModel(
            name=name or f"CI_SOGP_M{n_modes}_p{fixed_idx}",
            reducer=reducer,
            mode_factory=mode_factory,
            u=u,
            fixed_idx=fixed_idx,
        )

    @staticmethod
    def create_fieldwise_mogp(
        n_modes: int,
        u: np.ndarray,
        Q: int,
        fixed_idx: int,
        gp_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> FixedOutputModel:
        """FM: FieldwisePCA + per-mode MOGP-LCM (Q-1 correlated outputs)."""
        cfg = gp_config or {}
        q = Q - 1   # outputs per mode (non-fixed fields)

        def mode_factory():
            return MOGPLCMModeRegressor(
                output_dim=q,
                n_kernels=cfg.get("n_kernels", 2),
                rank=cfg.get("rank", None),
                n_restarts=cfg.get("n_restarts", 3),
                maxiter=cfg.get("maxiter", 100),
                noise_var=cfg.get("noise_var", 1e-3),
                seed=cfg.get("seed", None),
            )

        reducer = FieldwisePCA(n_modes=n_modes)
        return FixedOutputModel(
            name=name or f"FM_MOGPLCM_M{n_modes}_p{fixed_idx}",
            reducer=reducer,
            mode_factory=mode_factory,
            u=u,
            fixed_idx=fixed_idx,
        )

    @staticmethod
    def create_fieldwise_optimized(
        n_modes: int,
        u: np.ndarray,
        Q: int,
        fixed_indices: List[int],
        gp_config: Optional[Dict[str, Any]] = None,
    ) -> FieldwiseOptimizedModel:
        """FI: FieldwisePCA + IndepSOGP, one training for all fixed-output scenarios."""
        cfg = gp_config or {}
        return FieldwiseOptimizedModel(
            n_modes=n_modes,
            u=u,
            Q=Q,
            fixed_indices=fixed_indices,
            n_restarts=cfg.get("n_restarts", 3),
            maxiter=cfg.get("maxiter", 100),
            var_noise=cfg.get("noise_var", 1e-3),
            seed=cfg.get("seed", None),
        )

    # ------------------------------------------------------------------
    # Full benchmark suite
    # ------------------------------------------------------------------

    @staticmethod
    def create_benchmark_suite(
        n_modes_list: List[int],
        u: np.ndarray,
        Q: int,
        fixed_indices: List[int],
        gp_config: Optional[Dict[str, Any]] = None,
        gp_config_lmc: Optional[Dict[str, Any]] = None,
        gp_config_constrained: Optional[Dict[str, Any]] = None,
        include_rc: bool = True,
        include_ci: bool = True,
        include_fi: bool = True,
        include_fm: bool = True,
    ) -> Dict[str, Any]:
        """Create all model combinations.

        Returns a dict with two lists:
        - ``'standard'``: list of SurrogateModel objects (RC, CI, FM per n_modes per fixed_idx)
        - ``'optimized'``: list of FieldwiseOptimizedModel objects (FI, one per n_modes)

        Standard models are fit and evaluated one by one.
        Optimized models are fit once and generate Q scenarios internally.

        Parameters
        ----------
        gp_config            : shared fallback config for all models
        gp_config_lmc        : config for FM (MOGP-LCM); overrides gp_config for FM
        gp_config_constrained: config for RC (Constrained MOGP); overrides gp_config for RC
        """
        cfg_rc  = gp_config_constrained or gp_config
        cfg_fm  = gp_config_lmc         or gp_config
        cfg_ci  = gp_config
        cfg_fi  = gp_config

        standard  = []
        optimized = []

        for M in n_modes_list:
            if include_rc:
                standard.append(
                    ModelRegistry.create_rowwise_constrained(M, u, cfg_rc)
                )
            if include_ci:
                for p in fixed_indices:
                    standard.append(
                        ModelRegistry.create_colwise_independent(M, u, p, cfg_ci)
                    )
            if include_fm:
                for p in fixed_indices:
                    standard.append(
                        ModelRegistry.create_fieldwise_mogp(M, u, Q, p, cfg_fm)
                    )
            if include_fi:
                optimized.append(
                    ModelRegistry.create_fieldwise_optimized(
                        M, u, Q, fixed_indices, cfg_fi
                    )
                )

        return {"standard": standard, "optimized": optimized}

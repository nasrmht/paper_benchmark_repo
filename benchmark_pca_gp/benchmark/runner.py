"""BenchmarkRunner: orchestrates data generation, model fitting, and evaluation."""
import time
import traceback
import numpy as np
from typing import List, Optional, Dict, Any

from ..data.base import Dataset
from ..models.base import SurrogateModel
from ..models.fieldwise_optimized import FieldwiseOptimizedModel
from ..metrics.metrics import compute_all_metrics
from .storage import open_storage


class BenchmarkRunner:
    """Runs a full benchmark: data generation → fit → evaluate → store.

    Parameters
    ----------
    dataset       : Dataset instance (e.g. LotkaVolterraDataset)
    storage_path  : path to output Zarr archive
    n_train       : number of training samples
    n_total       : total samples (n_train + n_test)
    seed          : random seed for data generation and train/test split
    verbose       : print progress if True
    skip_existing : if True, skip models already in the storage
    """

    def __init__(
        self,
        dataset: Dataset,
        storage_path: str,
        n_train: int,
        n_total: int,
        seed: int = 42,
        verbose: bool = True,
        skip_existing: bool = False,
        store_predictions: bool = True,
    ):
        self.dataset = dataset
        self.storage = open_storage(storage_path, mode="a")
        self.n_train = n_train
        self.n_total = n_total
        self.seed = seed
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.store_predictions = store_predictions

        # Will be populated on first run
        self._X_train = None
        self._X_test  = None
        self._fields_train_orig  = None   # raw y-space (for ground-truth storage)
        self._fields_test_orig   = None   # raw y-space
        self._fields_train_c     = None   # z-space centred (for intermediate metrics)
        self._fields_test_c      = None   # z-space centred
        self._means_train        = None   # z-space means
        self._fields_train_model = None   # z-space NOT-centred (for model.fit)
        self._fields_test_model  = None   # z-space NOT-centred (for model.evaluate)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        if self._X_train is not None:
            return   # already done

        if self.verbose:
            print(f"[Runner] Generating {self.n_total} samples (seed={self.seed})…")

        X, fields = self.dataset.generate(self.n_total, self.seed)
        X_tr, X_te, f_tr, f_te = self.dataset.split_train_test(
            X, fields, self.n_train, self.seed
        )

        # Center in z-space: applies α+H transform (Case 3) when the dataset
        # overrides input_weights(), otherwise standard centering (Case 1/2).
        # X_train/X_test are passed so that per-sample α and H can be computed.
        f_tr_z, f_te_z, means_z = self.dataset.center(
            f_tr, f_te, X_train=X_tr, X_test=X_te
        )

        # z-space NOT-centred fields: models subtract means internally, so they
        # receive these and produce z-space predictions when means are added back.
        Q = self.dataset.n_outputs
        f_tr_model = [f_tr_z[k] + means_z[k] for k in range(Q)]
        f_te_model = [f_te_z[k] + means_z[k] for k in range(Q)]

        self._X_train            = X_tr
        self._X_test             = X_te
        self._fields_train_orig  = f_tr          # raw y-space (ground-truth storage)
        self._fields_test_orig   = f_te          # raw y-space
        self._fields_train_c     = f_tr_z        # z-space centred (intermediate metrics)
        self._fields_test_c      = f_te_z        # z-space centred
        self._means_train        = means_z       # z-space means
        self._fields_train_model = f_tr_model    # z-space NC → model.fit()
        self._fields_test_model  = f_te_model    # z-space NC → model.evaluate()

        # Save ground truth and config once.
        # Ground truth is stored in z-space NC so that the PCA comparison
        # analysis works in the same space as the models.
        self.storage.save_config({
            "dataset":  type(self.dataset).__name__,
            "n_train":  self.n_train,
            "n_test":   self.n_total - self.n_train,
            "n_total":  self.n_total,
            "seed":     self.seed,
            "n_outputs":   self.dataset.n_outputs,
            "input_dim":   self.dataset.input_dim,
            "constraint_u": self.dataset.constraint_vector.tolist(),
            "t_end":    getattr(self.dataset, "t_end", None),
            "dt":       getattr(self.dataset, "dt",    None),
        })
        self.storage.save_ground_truth(X_tr, X_te, f_tr_model, f_te_model)

        if self.verbose:
            print(f"[Runner] Data ready: {self.n_train} train, "
                  f"{self.n_total - self.n_train} test.")

    # ------------------------------------------------------------------
    # Running standard SurrogateModels
    # ------------------------------------------------------------------

    def _run_standard_model(self, model: SurrogateModel) -> None:
        name = model.name
        if self.skip_existing and self.storage.model_exists(name):
            if self.verbose:
                print(f"[Runner] Skipping {name} (already stored).")
            return

        if self.verbose:
            print(f"[Runner] Fitting  {name} …")

        t0 = time.time()
        try:
            model.fit(
                self._X_train,
                self._fields_train_model,
                self._means_train,
            )
        except Exception as e:
            print(f"[Runner] ERROR fitting {name}: {e}")
            traceback.print_exc()
            return

        fit_time = time.time() - t0
        if self.verbose:
            print(f"[Runner] Fitted   {name} in {fit_time:.1f}s  → evaluating…")

        t1 = time.time()
        try:
            result = model.evaluate(
                self._X_test,
                self._fields_test_model,
                self._means_train,
            )
        except Exception as e:
            print(f"[Runner] ERROR evaluating {name}: {e}")
            traceback.print_exc()
            return
        eval_time = time.time() - t1

        # Intermediate metrics (PCA + GP latent)
        intermediate = {}
        try:
            inter = model.intermediate_metrics(
                self._X_test,
                self._fields_test_c,
                self._fields_train_c,
                self._means_train,
            )
            intermediate = {
                k: v for k, v in inter.items()
                if not isinstance(v, dict)
            }
        except Exception:
            pass

        try:
            if "predicted_weights" in result["predictions"]:
                cum_pred = model.reducer.cumulative_prediction_error(
                    result["predictions"]["predicted_weights"],
                    self._fields_test_c,
                    self._means_train,
                    split_name="test",
                    fixed_idx=model.fixed_idx,
                    u=self.dataset.constraint_vector,
                )
                intermediate.update(cum_pred)
        except Exception as e:
            print(f"[Runner] ERROR computing cumulative prediction: {e}")
            traceback.print_exc()

        # Stack predictions into (N_test, Q, S)
        fields_mean = result["predictions"]["fields_mean"]
        fields_var  = result["predictions"]["fields_var"]
        N_test, S = fields_mean[0].shape
        pred_mean = np.stack(fields_mean, axis=1)   # (N_test, Q, S)
        pred_var  = np.stack(fields_var,  axis=1)   # (N_test, Q, S)

        # Build config metadata
        config = {
            "model_name":  name,
            "pca_type":    type(model.reducer).__name__,
            "gp_type":     "ConstrainedMOGP" if hasattr(model, "_mode_models") else
                            type(list(model._regressor.models)[0]).__name__
                            if hasattr(model, "_regressor") else "unknown",
            "n_modes":     model.reducer.n_modes,
            "fixed_idx":   model.fixed_idx if model.fixed_idx is not None else -1,
            "fit_time_s":  round(fit_time, 2),
            "eval_time_s": round(eval_time, 2),
        }

        # Normalisation params
        normalisation = {
            "means_train": self._means_train,
            "std_weights": model.get_std_weights()
                           if hasattr(model, "get_std_weights") else None,
        }

        # Final metrics (exclude raw predictions)
        metrics = {k: v for k, v in result.items()
                   if k != "predictions" and not isinstance(v, list)}

        self.storage.save_model_result(
            model_name=name,
            config=config,
            predictions_mean=pred_mean if self.store_predictions else None,
            predictions_var=pred_var   if self.store_predictions else None,
            metrics=metrics,
            intermediate=intermediate,
            normalisation=normalisation,
        )

        if self.verbose:
            q2_str = ", ".join(f"{v:.3f}" for v in result["q2"])
            print(f"[Runner] Saved    {name} | Q²=[{q2_str}] | "
                  f"constraint_max={result['constraint_max']:.2e}")

    # ------------------------------------------------------------------
    # Running FieldwiseOptimizedModels
    # ------------------------------------------------------------------

    def _run_optimized_model(self, model: FieldwiseOptimizedModel) -> None:
        base_name = f"FI_IndepSOGP_M{model.n_modes}"

        # Check if all scenarios are already stored
        if self.skip_existing and all(
            self.storage.model_exists(f"{base_name}_p{p}")
            for p in model.fixed_indices
        ):
            if self.verbose:
                print(f"[Runner] Skipping {base_name} scenarios (all stored).")
            return

        if self.verbose:
            print(f"[Runner] Fitting  {base_name} (all {model.Q} fields) …")

        t0 = time.time()
        try:
            model.fit(
                self._X_train,
                self._fields_train_model,
                self._means_train,
            )
        except Exception as e:
            print(f"[Runner] ERROR fitting {base_name}: {e}")
            traceback.print_exc()
            return

        fit_time = time.time() - t0
        if self.verbose:
            print(f"[Runner] Fitted   {base_name} in {fit_time:.1f}s.")

        # Latent metrics (same for all scenarios)
        latent_inter = {}
        try:
            lm = model.latent_metrics(self._X_test, self._fields_test_c)
            latent_inter = lm
        except Exception:
            pass

        # Evaluate each scenario
        for p in model.fixed_indices:
            scenario_name = f"{base_name}_p{p}"
            if self.skip_existing and self.storage.model_exists(scenario_name):
                continue

            scenario = model.get_scenario(p)
            t1 = time.time()
            try:
                result = scenario.evaluate(
                    self._X_test,
                    self._fields_test_model,
                    self._means_train,
                )
            except Exception as e:
                print(f"[Runner] ERROR evaluating {scenario_name}: {e}")
                traceback.print_exc()
                continue
            eval_time = time.time() - t1

            fields_mean = result["predictions"]["fields_mean"]
            fields_var  = result["predictions"]["fields_var"]
            pred_mean = np.stack(fields_mean, axis=1)
            pred_var  = np.stack(fields_var,  axis=1)

            # PCA intermediate metrics
            intermediate = {}
            try:
                pca_err_tr = model._reducer.reconstruction_error(
                    self._fields_train_c, self._means_train, "train"
                )
                pca_err_te = model._reducer.reconstruction_error(
                    self._fields_test_c, self._means_train, "test"
                )
                intermediate.update(pca_err_tr)
                intermediate.update(pca_err_te)
                intermediate.update(latent_inter)
                # Cumulative reconstruction k=1..M with deduction of field p
                cum = model._reducer.cumulative_reconstruction_error(
                    self._fields_test_c, self._means_train, "test",
                    fixed_idx=p, u=self.dataset.constraint_vector,
                )
                intermediate.update(cum)
            except Exception:
                pass

            try:
                if "predicted_weights" in result["predictions"]:
                    cum_pred = model._reducer.cumulative_prediction_error(
                        result["predictions"]["predicted_weights"],
                        self._fields_test_c,
                        self._means_train,
                        split_name="test",
                        fixed_idx=p,
                        u=self.dataset.constraint_vector,
                    )
                    intermediate.update(cum_pred)
            except Exception as e:
                print(f"[Runner] ERROR computing prediction cumulative: {e}")

            config = {
                "model_name":  scenario_name,
                "pca_type":    "FieldwisePCA",
                "gp_type":     "IndepSOGP",
                "n_modes":     model._reducer.n_modes,
                "fixed_idx":   p,
                "fit_time_s":  round(fit_time, 2),   # shared fit time
                "eval_time_s": round(eval_time, 2),
            }

            normalisation = {
                "means_train": self._means_train,
                "std_weights": model._all_std,
            }

            metrics = {k: v for k, v in result.items()
                       if k != "predictions" and not isinstance(v, list)}

            self.storage.save_model_result(
                model_name=scenario_name,
                config=config,
                predictions_mean=pred_mean if self.store_predictions else None,
                predictions_var=pred_var   if self.store_predictions else None,
                metrics=metrics,
                intermediate=intermediate,
                normalisation=normalisation,
            )

            if self.verbose:
                q2_str = ", ".join(f"{v:.3f}" for v in result["q2"])
                print(f"[Runner] Saved    {scenario_name} | Q²=[{q2_str}] | "
                      f"constraint_max={result['constraint_max']:.2e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        standard_models: Optional[List[SurrogateModel]] = None,
        optimized_models: Optional[List[FieldwiseOptimizedModel]] = None,
    ) -> None:
        """Run the full benchmark.

        Parameters
        ----------
        standard_models  : list of SurrogateModel instances (RC, CI, FM)
        optimized_models : list of FieldwiseOptimizedModel instances (FI)
        """
        self._prepare_data()

        for model in (standard_models or []):
            self._run_standard_model(model)

        for model in (optimized_models or []):
            self._run_optimized_model(model)

        # Release the joblib/loky workers created by the GP libraries
        # Avoids the 65 "leaked folder objects" and the delay at Python shutdown
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass

        if self.verbose:
            print("[Runner] Benchmark complete.")

    def run_from_suite(self, suite: Dict[str, list]) -> None:
        """Convenience wrapper for suites created by ModelRegistry."""
        self.run(
            standard_models=suite.get("standard", []),
            optimized_models=suite.get("optimized", []),
        )

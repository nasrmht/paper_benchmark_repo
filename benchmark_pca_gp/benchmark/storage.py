"""Storage backends for benchmark results.

Two backends are available:
  - ZarrBenchmarkStorage  : zarr 3.x, full data (fields + predictions)
  - PklBenchmarkStorage   : lightweight pickle, metrics only (recommended for repos)

Use open_storage(path) to auto-detect the backend from the file extension.
"""
import os
import pickle
import numpy as np
import zarr
from typing import List, Optional, Dict, Any


def _save_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    """Save a numpy array to a zarr group, compatible with zarr 3.x."""
    group[name] = data


def open_storage(path: str, mode: str = "a"):
    """Factory: return the right storage backend based on file extension.

    .pkl  → PklBenchmarkStorage  (lightweight, no zarr dependency at runtime)
    .zarr → ZarrBenchmarkStorage (full data, backward-compatible)
    """
    if path.endswith(".pkl"):
        return PklBenchmarkStorage(path, mode=mode)
    return ZarrBenchmarkStorage(path, mode=mode)


class ZarrBenchmarkStorage:
    """Stores benchmark data and results in a Zarr archive.

    Structure
    ---------
    results.zarr/
    ├── config/
    │   └── .attrs  {dataset_name, n_train, n_test, seed, ...}
    ├── ground_truth/
    │   ├── X_train        (N_train, d)
    │   ├── X_test         (N_test, d)
    │   ├── fields_train/
    │   │   ├── field_0    (N_train, S)
    │   │   └── ...
    │   └── fields_test/
    │       ├── field_0    (N_test, S)
    │       └── ...
    └── models/
        └── {model_name}/
            ├── .attrs  {pca_type, gp_type, n_modes, fixed_idx, ...}
            ├── intermediate/
            │   ├── pca_rrmse_train       (Q,)
            │   ├── pca_rrmse_test        (Q,)
            │   ├── latent_q2             (M, q) or absent
            │   └── latent_rmse           (M, q) or absent
            ├── normalisation/
            │   ├── means_train/field_i   (S,)
            │   └── std_weights           (M, q)
            └── final/
                ├── predictions_mean      (N_test, Q, S)
                ├── predictions_var       (N_test, Q, S)
                └── metrics/
                    ├── q2                (Q,)
                    ├── rrmse             (Q,)
                    ├── interval_width    (Q,)
                    ├── constraint_mean   (1,)
                    └── constraint_max    (1,)
    """

    def __init__(self, path: str, mode: str = "a"):
        self.path = path
        self.store = zarr.open(path, mode=mode)

    # ------------------------------------------------------------------
    # Dataset / ground truth
    # ------------------------------------------------------------------

    def save_config(self, config: Dict[str, Any]) -> None:
        grp = self.store.require_group("config")
        grp.attrs.update(config)

    def save_ground_truth(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        fields_train: List[np.ndarray],
        fields_test:  List[np.ndarray],
    ) -> None:
        gt = self.store.require_group("ground_truth")
        _save_array(gt, "X_train", X_train)
        _save_array(gt, "X_test",  X_test)

        tr = gt.require_group("fields_train")
        te = gt.require_group("fields_test")
        for i, (f_tr, f_te) in enumerate(zip(fields_train, fields_test)):
            _save_array(tr, f"field_{i}", f_tr)
            _save_array(te, f"field_{i}", f_te)

    def load_ground_truth(self) -> Dict[str, Any]:
        gt = self.store["ground_truth"]
        Q = sum(1 for k in gt["fields_train"] if k.startswith("field_"))
        return {
            "X_train":      gt["X_train"][:],
            "X_test":       gt["X_test"][:],
            "fields_train": [gt["fields_train"][f"field_{i}"][:] for i in range(Q)],
            "fields_test":  [gt["fields_test"][f"field_{i}"][:] for i in range(Q)],
        }

    # ------------------------------------------------------------------
    # Model results
    # ------------------------------------------------------------------

    def save_model_result(
        self,
        model_name: str,
        config: Dict[str, Any],
        predictions_mean: Optional[np.ndarray],  # (N_test, Q, S) or None
        predictions_var:  Optional[np.ndarray],  # (N_test, Q, S) or None
        metrics: Dict[str, Any],
        intermediate: Optional[Dict[str, Any]] = None,
        normalisation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save all results for one model.

        Pass ``predictions_mean=None`` and ``predictions_var=None`` to skip
        storing the raw prediction arrays (useful for large S to save disk).
        """
        grp = self.store.require_group(f"models/{model_name}")
        grp.attrs.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in config.items()})

        # Predictions (optional)
        final = grp.require_group("final")
        if predictions_mean is not None:
            _save_array(final, "predictions_mean", predictions_mean)
        if predictions_var is not None:
            _save_array(final, "predictions_var",  predictions_var)

        # Metrics
        met = final.require_group("metrics")
        for k, v in metrics.items():
            val = np.atleast_1d(np.asarray(v, dtype=float))
            _save_array(met, k, val)

        # Intermediate metrics
        if intermediate:
            inter = grp.require_group("intermediate")
            for k, v in intermediate.items():
                if v is not None:
                    _save_array(inter, k, np.asarray(v, dtype=float))

        # Normalisation parameters
        if normalisation:
            norm = grp.require_group("normalisation")
            means_grp = norm.require_group("means_train")
            for i, m in enumerate(normalisation.get("means_train", [])):
                _save_array(means_grp, f"field_{i}", np.asarray(m))
            if "std_weights" in normalisation and normalisation["std_weights"] is not None:
                _save_array(norm, "std_weights", np.asarray(normalisation["std_weights"]))

    def load_model_result(self, model_name: str) -> Dict[str, Any]:
        """Load all saved data for a model (config + metrics + intermediate + predictions)."""
        grp   = self.store[f"models/{model_name}"]
        final = grp["final"]
        met   = final["metrics"]

        result = {
            "config":  dict(grp.attrs),
            "metrics": {k: met[k][:] for k in met},
            "predictions_mean": final["predictions_mean"][:] if "predictions_mean" in final else None,
            "predictions_var":  final["predictions_var"][:] if "predictions_var" in final else None,
        }

        if "intermediate" in grp:
            inter = grp["intermediate"]
            result["intermediate"] = {k: inter[k][:] for k in inter}

        return result

    def load_model_result_light(self, model_name: str) -> Dict[str, Any]:
        """Loads config + metrics + intermediate WITHOUT predictions.

        Much faster than load_model_result() for summaries and
        DataFrames that do not need the arrays (N_test, Q, S).
        """
        grp = self.store[f"models/{model_name}"]
        met = grp["final"]["metrics"]

        result = {
            "config":  dict(grp.attrs),
            "metrics": {k: met[k][:] for k in met},
        }

        if "intermediate" in grp:
            inter = grp["intermediate"]
            result["intermediate"] = {k: inter[k][:] for k in inter}

        return result

    def list_models(self) -> List[str]:
        if "models" not in self.store:
            return []
        return list(self.store["models"].keys())

    def model_exists(self, model_name: str) -> bool:
        return f"models/{model_name}" in self.store

    def get_n_outputs(self) -> int:
        """Return Q (number of outputs). Reads from config attrs or from metrics."""
        cfg = dict(self.store.get("config", {}).attrs) if "config" in self.store else {}
        if "n_outputs" in cfg:
            return int(cfg["n_outputs"])
        # Fallback: infer from first model's q2 metric
        models = self.list_models()
        if models:
            res = self.load_model_result_light(models[0])
            return len(res["metrics"].get("q2", []))
        return 0


# ---------------------------------------------------------------------------
# PklBenchmarkStorage — lightweight pickle backend
# ---------------------------------------------------------------------------

class PklBenchmarkStorage:
    """Stores benchmark results in a single pickle file.

    Only metrics, intermediate metrics, and a few config values are kept.
    Raw fields (S-dimensional arrays) and prediction arrays are NOT stored,
    making the file very compact (typically a few KB per seed).

    Interface is identical to ZarrBenchmarkStorage.load_model_result_light().
    """

    def __init__(self, path: str, mode: str = "a"):
        self.path = path
        self._data: Dict[str, Any] = {}
        if mode in ("r", "a") and os.path.exists(path):
            with open(path, "rb") as f:
                self._data = pickle.load(f)

    # ------------------------------------------------------------------

    def _flush(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_config(self, config: Dict[str, Any]) -> None:
        self._data["config"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in config.items()
        }
        self._flush()

    def save_ground_truth(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        fields_train,   # not stored (large)
        fields_test,    # not stored (large)
    ) -> None:
        self._data["ground_truth"] = {
            "X_train": X_train,
            "X_test":  X_test,
        }
        self._flush()

    def load_ground_truth(self) -> Dict[str, Any]:
        gt = self._data.get("ground_truth", {})
        return {
            "X_train":      gt.get("X_train"),
            "X_test":       gt.get("X_test"),
            "fields_train": None,   # not stored in lightweight format
            "fields_test":  None,
        }

    def save_model_result(
        self,
        model_name: str,
        config: Dict[str, Any],
        predictions_mean,   # ignored (large)
        predictions_var,    # ignored (large)
        metrics: Dict[str, Any],
        intermediate: Optional[Dict[str, Any]] = None,
        normalisation: Optional[Dict[str, Any]] = None,
    ) -> None:
        if "models" not in self._data:
            self._data["models"] = {}

        cfg_stored = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in config.items()
        }
        model_data: Dict[str, Any] = {
            "config":  cfg_stored,
            "metrics": {k: np.atleast_1d(np.asarray(v, dtype=float))
                        for k, v in metrics.items()},
        }
        if intermediate:
            model_data["intermediate"] = {
                k: np.asarray(v, dtype=float)
                for k, v in intermediate.items() if v is not None
            }
        # Store only std_weights from normalisation (needed by analysis scripts)
        if normalisation and normalisation.get("std_weights") is not None:
            model_data["normalisation"] = {
                "std_weights": np.asarray(normalisation["std_weights"])
            }

        self._data["models"][model_name] = model_data
        self._flush()

    def load_model_result_light(self, model_name: str) -> Dict[str, Any]:
        model_data = self._data["models"][model_name]
        return {
            "config":       model_data.get("config", {}),
            "metrics":      model_data.get("metrics", {}),
            "intermediate": model_data.get("intermediate", {}),
        }

    def load_model_result(self, model_name: str) -> Dict[str, Any]:
        """Same as light: predictions not stored in pkl format."""
        result = self.load_model_result_light(model_name)
        result["predictions_mean"] = None
        result["predictions_var"]  = None
        return result

    def list_models(self) -> List[str]:
        return list(self._data.get("models", {}).keys())

    def model_exists(self, model_name: str) -> bool:
        return model_name in self._data.get("models", {})

    def get_n_outputs(self) -> int:
        cfg = self._data.get("config", {})
        if "n_outputs" in cfg:
            return int(cfg["n_outputs"])
        models = self.list_models()
        if models:
            res = self.load_model_result_light(models[0])
            return len(res["metrics"].get("q2", []))
        return 0

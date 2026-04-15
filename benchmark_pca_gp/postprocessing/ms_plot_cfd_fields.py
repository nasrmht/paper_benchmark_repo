"""
Visualize predicted vs simulated CFD fields for the RC model.

Outputs a 3x4 subplot figure:
- rows: tau11, tau22, k
- columns: Simulated, Predicted, Std Dev, SRSE (Spatial Relative Squared Error)

Usage
-----
    python ms_plot_cfd_fields.py --sample 0 --n_modes 8 --prefix results_n=50_cfd --seed 51
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_pca_gp.benchmark.storage import ZarrBenchmarkStorage

_DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Constrained_MOGP", "Application", "CFD", "cfd_diffuseur",
)

FIELD_NAMES = [r"$\tau_{11}$", r"$\tau_{22}$", r"$k$"]
PRED_FIELD_NAMES = [r"$\hat{\tau}_{11}$", r"$\hat{\tau}_{22}$", r"$\hat{k}$"]
ERR_NAMES = [
    r"SRSE($\hat{\tau}_{11}, \tau_{11}$)",
    r"SRSE($\hat{\tau}_{22}, \tau_{22}$)",
    r"SRSE($\hat{k}, k$)",
]

def parse_args():
    p = argparse.ArgumentParser(description="Plot CFD fields for RC model predictions.")
    p.add_argument("--prefix", default="results_n=50_cfd", help="Prefix of the Zarr file")
    p.add_argument("--seed", type=int, default=52, help="Seed used for Zarr file")
    p.add_argument("--n_modes", type=int, default=8, help="Number of latent PCA modes (M) for RC model")
    p.add_argument("--sample", type=int, default=1, help="Test sample index (0 to 99)")
    p.add_argument("--outdir", default="figures_cfd", help="Output directory for plots")
    p.add_argument("--data_root", default=None, help="Root for CFD data (to load mesh coords)")
    return p.parse_args()


def load_mesh_coordinates(data_root: str):
    """
    Extract x, y from the first column of the 'k' field training file.
    The variables k_*.npy have shape (S, 3) where column 0 is x and column 1 is y.
    """
    data_root = "/home/catC/mn279127/Documents/thèse_mahamat/code/Constrained_MOGP/Application/CFD/cfd_diffuseur"
    k_file = os.path.join(data_root, "Datasets_train_16072025", "k", "k_1.npy")
    if not os.path.exists(k_file):
        raise FileNotFoundError(f"Could not find coordinates source: {k_file}")
    
    k_data = np.load(k_file, mmap_mode="r")
    x = k_data[:, 0]
    y = k_data[:, 1]
    return x, y


def plot_simulated_fields_only(x, y, fields_test, sample_idx, outdir, prefix, seed, n_modes):
    """
    Plots only the 3 simulated fields (tau11, tau22, k) in 3 rows with figsize (12, 6).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)
    cmap = "jet"
    
    print(f"Generating simulated only plot for Sample Index: {sample_idx}")
    for i in range(3):
        field_sim = fields_test[i][sample_idx]
        ax = axes[i]
        im = ax.scatter(x, y, c=field_sim, cmap=cmap, s=4, rasterized=True)
        ax.set_title("Simulated " + FIELD_NAMES[i], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, location="right")
        
    plt.tight_layout()
    
    suffix = f"s{sample_idx}_RC_M{n_modes}_seed{seed}"
    out_name = f"simulated_only_{suffix}.pdf"
    out_path = os.path.join(outdir, out_name)
    plt.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    print(f"Saved simulated only plot to: {out_path}")
    
    out_path_png = os.path.join(outdir, f"simulated_only_{suffix}.png")
    plt.savefig(out_path_png, format="png", dpi=150, bbox_inches="tight")
    print(f"Saved simulated only PNG plot to: {out_path_png}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    data_root = args.data_root or _DEFAULT_DATA_ROOT
    zarr_path = f"{args.prefix}_seed{args.seed}.zarr"
    
    print(f"Loading from storage: {zarr_path}")
    if not os.path.exists(zarr_path):
        print(f"Error: {zarr_path} does not exist. Are you in the right directory?")
        sys.exit(1)
        
    storage = ZarrBenchmarkStorage(zarr_path, mode="r")
    
    # Load coordinates
    print("Loading mesh coordinates...")
    x, y = load_mesh_coordinates(data_root)
    
    # Load model
    model_name = f"RC_ConstMOGP_M{args.n_modes}"
    
    if not storage.model_exists(model_name):
        print(f"Error: Model '{model_name}' not found in {zarr_path}.")
        available = [m for m in storage.list_models() if m.startswith("RC")]
        print(f"Available RC models: {available}")
        sys.exit(1)
        
    res = storage.load_model_result(model_name)
    predictions_mean = res.get("predictions_mean")
    predictions_var = res.get("predictions_var")
    if predictions_mean is None:
        print(f"Error: No prediction arrays saved in the Zarr archive for {model_name}.")
        print("Make sure 'run_cfd.py' was run without '--no_predictions' flag.")
        sys.exit(1)
        
    print("Loading test data...")
    gt = storage.load_ground_truth()
    fields_test = gt["fields_test"]  # shape: list of 3 arrays of (100, S)
    
    sample_idx = args.sample
    if not (0 <= sample_idx < len(fields_test[0])):
        print(f"Invalid sample idx {sample_idx}. Total test samples: {len(fields_test[0])}")
        sys.exit(1)
    
    # Prepare figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 8), sharex=True, sharey=True)
    cmap = "jet" # "turbo" # "inferno" # "viridis"
    
    print(f"Generating plot for Sample Index: {sample_idx}")
    for i in range(3):
        field_sim = fields_test[i][sample_idx]
        field_pred = predictions_mean[sample_idx, i]
        
        # Simulated
        ax = axes[i, 0]
        im = ax.scatter(x, y, c=field_sim, cmap=cmap, s=4, rasterized=True)
        ax.set_title("Simulated " + FIELD_NAMES[i], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, location="right")
        
        # Predicted
        ax = axes[i, 1]
        im = ax.scatter(x, y, c=field_pred, cmap=cmap, s=4, rasterized=True)
        ax.set_title("Predicted " + PRED_FIELD_NAMES[i], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, location="right")
        
        # Std Dev
        field_var = predictions_var[sample_idx, i] if predictions_var is not None else np.zeros_like(field_sim)
        field_std = np.sqrt(np.maximum(field_var, 0.0))
        ax = axes[i, 2]
        im = ax.scatter(x, y, c=field_std, cmap=cmap, s=4, rasterized=True)
        ax.set_title("Std Dev " + PRED_FIELD_NAMES[i], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, location="right")
        
        # SRSE
        ax = axes[i, 3]
        norm_inf_sq = np.linalg.norm(field_sim, ord=np.inf)**2
        # Avoid division by 0
        norm_inf_sq = max(norm_inf_sq, 1e-12)
        srse = ((field_pred - field_sim)**2) / norm_inf_sq
        
        im = ax.scatter(x, y, c=srse, cmap=cmap, s=4, rasterized=True)
        ax.set_title(ERR_NAMES[i], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, location="right")
        
    plt.tight_layout()
    
    suffix = f"s{sample_idx}_RC_M{args.n_modes}_seed{args.seed}"
    out_name = f"cfd_fields_{suffix}.pdf"
    out_path = os.path.join(args.outdir, out_name)
    plt.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
    
    # Save a PNG as well to be easy to view
    out_path_png = os.path.join(args.outdir, f"cfd_fields_{suffix}.png")
    plt.savefig(out_path_png, format="png", dpi=150, bbox_inches="tight")
    #plt.show()
    print(f"Saved PNG plot to: {out_path_png}")
    
    # Plot only the simulated fields in 3 rows
    plot_simulated_fields_only(x, y, fields_test, sample_idx, args.outdir, args.prefix, args.seed, args.n_modes)

    
if __name__ == "__main__":
    main()

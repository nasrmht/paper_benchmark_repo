import numpy as np
import pickle
import sys
import os
import argparse
from sklearn.metrics import r2_score, mean_squared_error

# Add root directory to path to allow imports of our packages
# Assumes structure: .../Constraint/Benchmark_complet_cluster.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# --- MOGP Imports ---
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LMCKernel import LMCKernel
from LcGP.mogp.kernels.ConstrainedLMCKernel import LMCKernelConstrained
from LcGP.mogp.kernels.Kernel import Matern52Kernel as MaternKernel_MO

# --- Single Output GP Imports ---
from LcGP.sogp.core import so_GPRegression
# Using Matern52 for Independent to be consistent with MO kernel
from LcGP.sogp.kernels.Kernel import MaternKernel as MaternKernel_SO

from scipy.stats.qmc import LatinHypercube 

# ==========================================
# 1. Data Generation (Identical to benchmark_cgp_lcm.py)
# ==========================================
def scaled_ishigami(X_unit):
    X = X_unit * 2 * np.pi - np.pi
    return np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + 0.1 * (X[:, 2]**4) * np.sin(X[:, 0])

def scaled_branin(X_unit):
    x1 = 15 * X_unit[:, 0] - 5
    x2 = 15 * X_unit[:, 1]
    a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return a * (x2 - b*x1**2 + c*x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

def generate_complex_data(n, noise=0.0, seed=42):
    rng = np.random.RandomState(seed)
    lhs = LatinHypercube(d=3, seed=seed)
    X = lhs.random(n=n)
    mean_y1 = 3.46
    mean_y2 = 54.8
    std_y1 = 3.75
    std_y2 = 50.75
    y1 = (scaled_ishigami(X)-mean_y1)/std_y1
    y2 = (scaled_branin(X)-mean_y2)/std_y2
    y3 = -y1 - y2 
    Y = np.vstack([y1, y2, y3]).T
    if noise > 0:
        Y += rng.normal(0, noise, Y.shape)
    return X, Y

# ==========================================
# 2. Metrics (metrics + coverage)
# ==========================================
def compute_single_metrics(y_true, y_pred, variance):
    q2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_val = np.std(y_true)
    rrmse = rmse #/ std_val if std_val > 1e-9 else 0.0
    sigma = np.sqrt(np.abs(variance)) 
    
    # Interval Length (95% => +/- 1.96 sigma)
    interval_len_mean = np.mean(2 * 1.96 * sigma)
    
    # Marginal Coverage Rate (90% interval => 5% to 95%)
    lower = y_pred - 1.645 * sigma
    upper = y_pred + 1.645 * sigma
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage_rate = np.mean(in_interval)
    
    return {'q2': q2, 'rrmse': rrmse, 'interval_len': interval_len_mean, 'coverage_rate': coverage_rate}

# ==========================================
# 3. Core Processing Logic
# ==========================================
def process_single_seed(seed_idx, n_train_list, n_test, noise_level):
    print(f"--> Processing Seed {seed_idx} started")
    
    # Result structure
    # Scenarios:
    # 1. 'independent_deduced_f1': Train Indep(f2), Indep(f3) -> Deduce f1
    # 2. 'lcm_deduced_f1': Train LCM(f2,f3) -> Deduce f1
    # 3. 'mogp_constrained': Train MOGP(f1,f2,f3)
    # Note: We can implement permutations for Indep/LCM if desired.
    # For "Benchmark_complet", let's include all 3 permutations for Indep and LCM to be thorough.
    
    scenarios = [
        'mogp_constrained',
        'lcm_deduced_f1', 'lcm_deduced_f2', 'lcm_deduced_f3',
        'indep_deduced_f1', 'indep_deduced_f2', 'indep_deduced_f3'
    ]
    
    metrics = ['q2', 'rrmse', 'interval_len', 'coverage_rate']
    
    local_results = {n: {s: {out: {m: [] for m in metrics} for out in range(3)} for s in scenarios} for n in n_train_list}

    # Test Data (Fixed for this seed)
    X_test, Y_test = generate_complex_data(n_test, noise=0.0, seed=999+seed_idx)
    
    for n in n_train_list:
        print(f"  Seed {seed_idx} - Training N={n}...")
        X_train, Y_train = generate_complex_data(n, noise=noise_level, seed=seed_idx*100 + n)
        
        # Normalization (Per column)
        mean_y = np.mean(Y_train, axis=0)
        std_y_col = np.std(Y_train, axis=0)
        Y_tr_centered = (Y_train - mean_y) / std_y_col
        
        # -----------------------------------------------
        # A. Independent GPs (Indep -> Deduce)
        # -----------------------------------------------
        indep_configs = {
            'indep_deduced_f1': {'target_idx': 0, 'train_cols': [1, 2]},
            'indep_deduced_f2': {'target_idx': 1, 'train_cols': [0, 2]},
            'indep_deduced_f3': {'target_idx': 2, 'train_cols': [0, 1]}
        }
        
        # 1. Train all 3 outputs independently once
        single_gp_mus = {} 
        single_gp_vars = {} 
        
        for c in range(3):
            # Train Independent GPs on column c
            # Fix: Use MaternKernel with nu=2.5 and explicit vector lengthscale for ARD
            k_so = MaternKernel_SO(length_scale=np.ones(3), nu=2.5) 
                
            model_so = so_GPRegression(kernel=k_so, var_noise=1e-6, use_kernel_grad=True, noisy_data=True, parallel=True)
            model_so.fit(X_train, Y_tr_centered[:, c],multi_start=True, n_start=30, seed=seed_idx)
                
            mu_c, var_c = model_so.predict(X_test)
                
            # Denormalize & Store
            single_gp_mus[c] = mu_c * std_y_col[c] + mean_y[c]
            single_gp_vars[c] = var_c * (std_y_col[c]**2)

        # 2. Iterate scenarios (deduction) using stored predictions
        for scen_name, conf in indep_configs.items():
            target_idx = conf['target_idx']
            cols = conf['train_cols']
            
            mus = np.zeros((n_test, 3))
            vars_ = np.zeros((n_test, 3))
            
            # Fill source columns from pre-calculated models
            for c in cols:
                mus[:, c] = single_gp_mus[c]
                vars_[:, c] = single_gp_vars[c]
            
            # Deduce Target
            mus[:, target_idx] = - np.sum(mus[:, cols], axis=1)
            # Var(Sum Indep) = Sum Vars (approx, ignoring correlation which SOGP ignores)
            vars_[:, target_idx] = np.sum(vars_[:, cols], axis=1)
            
            # Compute Metrics
            for i in range(3):
                met = compute_single_metrics(Y_test[:, i], mus[:, i], vars_[:, i])
                for k, v in met.items():
                    local_results[n][scen_name][i][k].append(v)


        # -----------------------------------------------
        # B. LCM (Correlated -> Deduce)
        # -----------------------------------------------
        lcm_configs = {
            'lcm_deduced_f1': {'target_idx': 0, 'train_cols': [1, 2]},
            'lcm_deduced_f2': {'target_idx': 1, 'train_cols': [0, 2]},
            'lcm_deduced_f3': {'target_idx': 2, 'train_cols': [0, 1]}
        }
        
        for scen_name, conf in lcm_configs.items():
            target_idx = conf['target_idx']
            cols = conf['train_cols']
            Y_sub = Y_tr_centered[:, cols]
            print("train lcm :")
            k1, k2 = MaternKernel_MO(input_dim=3), MaternKernel_MO(input_dim=3)
            lmc_2out = LMCKernel(base_kernels=[k1, k2], output_dim=2, rank=[1, 1], seed=seed_idx)
            model_lcm = MOGPR(kernel=lmc_2out, noise_variance=1e-6, use_efficient_lik=False)
            model_lcm.fit(X_train, Y_sub, n_restarts=50, maxiter=200, seed=seed_idx, verbose=False, use_init_pca=True)
            print("end training lcm")
            mu_raw, cov_raw = model_lcm.predict(X_test, return_cov=True, full_cov=True)
            
            # Extract
            mu_A, mu_B = mu_raw[:, 0], mu_raw[:, 1]
            var_A = np.diag(cov_raw)[:n_test]
            var_B = np.diag(cov_raw)[n_test:]
            cov_AB = np.diag(cov_raw[:n_test, n_test:])
            
            mus = np.zeros((n_test, 3))
            vars_ = np.zeros((n_test, 3))
            
            # De-normalize Sources
            mus[:, cols[0]] = mu_A * std_y_col[cols[0]] + mean_y[cols[0]]
            vars_[:, cols[0]] = var_A * (std_y_col[cols[0]]**2)
            mus[:, cols[1]] = mu_B * std_y_col[cols[1]] + mean_y[cols[1]]
            vars_[:, cols[1]] = var_B * (std_y_col[cols[1]]**2)
            
            # Deduce
            mus[:, target_idx] = - mus[:, cols[0]] - mus[:, cols[1]]
            # Var(Target) = Var(A) + Var(B) + 2 Cov(A,B)
            denom = vars_[:, cols[0]] + vars_[:, cols[1]] + 2 * cov_AB * std_y_col[cols[0]] * std_y_col[cols[1]]
            vars_[:, target_idx] = denom
            
            for i in range(3):
                met = compute_single_metrics(Y_test[:, i], mus[:, i], vars_[:, i])
                for k, v in met.items():
                    local_results[n][scen_name][i][k].append(v)
                    
        # -----------------------------------------------
        # C. Constrained MOGP
        # -----------------------------------------------
        k1_c, k2_c, k3_c = MaternKernel_MO(input_dim=3), MaternKernel_MO(input_dim=3), MaternKernel_MO(input_dim=3)
        lmc_constrained = LMCKernelConstrained(
            base_kernels=[k1_c, k2_c, k3_c], output_dim=3, 
            u_vector=np.ones(3)*std_y_col, rank=[1, 1, 1], seed=seed_idx
        )
        print("train constrained")
        model_const = MOGPR(kernel=lmc_constrained, noise_variance=1e-6, use_efficient_lik=False)
        model_const.fit(X_train, Y_tr_centered, n_restarts=50, maxiter=200, seed=seed_idx, verbose=False, use_init_pca=True)
        print("end training constrained")
        mu_mo, cov_mo = model_const.predict(X_test, return_cov=True, full_cov=True)
        # De-normalize
        Y_pred_mo = mu_mo * std_y_col + mean_y
        diag_vars_mo = np.diag(cov_mo).reshape(n_test, 3, order='F')
        Vars_mo = diag_vars_mo * (std_y_col**2)
        
        for i in range(3):
            met = compute_single_metrics(Y_test[:, i], Y_pred_mo[:, i], Vars_mo[:, i])
            for k, v in met.items():
                local_results[n]['mogp_constrained'][i][k].append(v)

    return local_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Benchmark for a single seed')
    parser.add_argument('--seed', type=int, default=11, help='Seed index to process')
    parser.add_argument('--n_train', nargs='+', type=int, default=[20, 50, 100], help='List of N_train sizes')
    parser.add_argument('--save_dir', type=str, default='results_complet', help='Directory to save results')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = process_single_seed(args.seed, args.n_train, n_test=500, noise_level=0.0)
    
    save_path = os.path.join(args.save_dir, f"benchmark_complet_seed={args.seed}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {save_path}")

import sys; sys.path.insert(0, '..')
import matplotlib.pyplot as plt
from benchmark_pca_gp.postprocessing.analysis import MultiSeedAnalyzer
List_results = [f"results_n=10_lv_seed{i}.zarr" for i in range(1, 3)]
ana = MultiSeedAnalyzer(List_results)
df_pm = ana.to_per_mode_dataframe()
print('Modèles avec métriques latentes:')
for name in sorted(df_pm.model_name.unique()):
    sub = df_pm[df_pm.model_name == name]
    print(f'  {name}: modes={sorted(sub.mode_idx.unique())}  seeds={sorted(sub.seed.unique())}')
print()
ana.print_summary()

df = ana.to_dataframe()           # (seed, model, metrics)
df_pm = ana.to_per_mode_dataframe()  # (seed, model, mode_idx, mean_latent_q2, mean_latent_rmse)

fig1 = ana.plot_comparison_violin_q2(n_modes=10)      # violin Q² : RC vs CI | FI | FM
fig2 = ana.plot_comparison_rrmse_per_mode(n_modes=10) # RRMSE/mode : RC vs CI | FI | FM
#fig = ana.plot_final_metrics_vs_modes(output_path="metrics_vs_modes.png")

#fig3 = ana.plot_cumulative_pca_quality(n_modes=10, output_path="cumulative_pca.png")
#ana = MultiSeedAnalyzer("results_lv_seed*.zarr")
fig4 = ana.plot_final_metrics_vs_modes(n_modes=10, model_types=["RC", "FI", "CI", 'FM'], output_path="metrics_vs_modes_n2=15.png")
plt.show()


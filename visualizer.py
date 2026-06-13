import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================================================
# 1. Load data and clean rows with failed solutions (error prevention)
# =========================================================================
try:
    df = pd.read_csv('benchmark_results.csv')
    # Filter out rows that do not have OR-Tools results (e.g., due to timeout)
    df_plot = df.dropna(subset=['ortools_dist', 'improvement', 'veh_saved']).copy()
    # Ensure the saved vehicles column is float type for accurate mean calculation
    df_plot['veh_saved'] = df_plot['veh_saved'].astype(float)
except FileNotFoundError:
    print("Error: 'benchmark_results.csv' not found. Please ensure it is in the same directory.")
    exit()

# =========================================================================
# 2. Configure academic/business high-end chart style
# =========================================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 18
})

# Create a 3x2 subplot grid (The Expanded Dashboard)
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle('MDCVRP Dual-Year Benchmark Analytics: Comprehensive Dashboard', fontweight='bold', y=0.98)

# =========================================================================
# Figure 1 (Row 0, Col 0): INCOM 2024 Distance Comparison
# =========================================================================
df_incom = df_plot[df_plot['dataset'] == 'incom2024'].sort_values(by='nodes').reset_index(drop=True)
axes[0, 0].plot(df_incom.index, df_incom['greedy_dist'], label='Greedy Baseline', color='#ff7f0e', linewidth=2, alpha=0.8)
axes[0, 0].plot(df_incom.index, df_incom['ortools_dist'], label='OR-Tools (Optimized)', color='#1f77b4', linewidth=2, alpha=0.9)
axes[0, 0].fill_between(df_incom.index, df_incom['ortools_dist'], df_incom['greedy_dist'], color='green', alpha=0.1, label='Distance Saved')

axes[0, 0].set_title('INCOM 2024: Routing Distance Comparison', fontweight='bold', pad=10)
axes[0, 0].set_xlabel('Instance Index (Sorted by Problem Scale / Nodes)')
axes[0, 0].set_ylabel('Total Mileage (km)')
axes[0, 0].legend()

# =========================================================================
# Figure 2 (Row 0, Col 1): MIM 2025 Distance Comparison
# =========================================================================
df_mim = df_plot[df_plot['dataset'] == 'mim2025'].sort_values(by='nodes').reset_index(drop=True)
axes[0, 1].plot(df_mim.index, df_mim['greedy_dist'], label='Greedy Baseline', color='#ff7f0e', linewidth=2, alpha=0.8)
axes[0, 1].plot(df_mim.index, df_mim['ortools_dist'], label='OR-Tools (Optimized)', color='#1f77b4', linewidth=2, alpha=0.9)
axes[0, 1].fill_between(df_mim.index, df_mim['ortools_dist'], df_mim['greedy_dist'], color='green', alpha=0.1, label='Distance Saved')

axes[0, 1].set_title('MIM 2025: Routing Distance Comparison', fontweight='bold', pad=10)
axes[0, 1].set_xlabel('Instance Index (Sorted by Problem Scale / Nodes)')
axes[0, 1].set_ylabel('Total Mileage (km)')
axes[0, 1].legend()

# =========================================================================
# Figure 3 (Row 1, Col 0): Optimization Rate Distribution Histogram
# =========================================================================
sns.histplot(data=df_plot, x='improvement', kde=True, ax=axes[1, 0], color='#1f77b4', bins=20)
mean_imp = df_plot['improvement'].mean()
axes[1, 0].axvline(mean_imp, color='red', linestyle='--', linewidth=2.5, label=f'Mean Improvement: +{mean_imp:.2f}%')

axes[1, 0].set_title('Distribution of Optimization Improvement Rate (%)', fontweight='bold', pad=10)
axes[1, 0].set_xlabel('Total Distance Saved Rate (%)')
axes[1, 0].set_ylabel('Count of Instances')
axes[1, 0].legend()

# =========================================================================
# Figure 4 (Row 1, Col 1): Problem Scale vs. Optimization Rate Scatter Plot
# =========================================================================
# Using style and markers to prevent overlapping dots from looking like a third color
sns.scatterplot(data=df_plot, x='nodes', y='improvement', hue='dataset', style='dataset',
                markers=['o', 's'], palette='Set1', ax=axes[1, 1], s=70, alpha=0.75)
sns.regplot(data=df_plot, x='nodes', y='improvement', scatter=False, ax=axes[1, 1], color='red', line_kws={'label': 'Performance Trend Line'})

axes[1, 1].set_title('Optimization Leverage vs. Problem Dimension (Nodes Count)', fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Problem Scale (Nodes Count)')
axes[1, 1].set_ylabel('Transport Distance Saved Rate (%)')
axes[1, 1].legend()

# =========================================================================
# Figure 5 (Row 2, Col 0): Greedy Baseline vs. OR-Tools Scatter Plot
# =========================================================================
# Using style and markers to distinguish datasets clearly even when overlapping
sns.scatterplot(data=df_plot, x='greedy_dist', y='ortools_dist', hue='dataset', style='dataset',
                markers=['o', 's'], palette='Set1', ax=axes[2, 0], s=70, alpha=0.75)
sns.regplot(data=df_plot, x='greedy_dist', y='ortools_dist', scatter=False, ax=axes[2, 0], color='red', line_kws={'label': 'Actual Algorithm Trend'})

max_val = max(df_plot['greedy_dist'].max(), df_plot['ortools_dist'].max())
min_val = min(df_plot['greedy_dist'].min(), df_plot['ortools_dist'].min())
axes[2, 0].plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y=x Baseline (Zero Improvement)')

axes[2, 0].set_title('Routing Performance Comparison: Greedy vs. OR-Tools', fontweight='bold', pad=10)
axes[2, 0].set_xlabel('Greedy Baseline Distance (km)')
axes[2, 0].set_ylabel('Google OR-Tools Distance (km)')
axes[2, 0].legend()

# =========================================================================
# Figure 6 (Row 2, Col 1): Saved Vehicles Distribution Histogram
# =========================================================================
sns.histplot(data=df_plot, x='veh_saved', discrete=True, ax=axes[2, 1], color='#2ca02c', alpha=0.8)
mean_veh = df_plot['veh_saved'].mean()
axes[2, 1].axvline(mean_veh, color='red', linestyle='--', linewidth=2.5, label=f'Mean Vehicles Saved: +{mean_veh:.2f}')

# Force X-axis to display only integers
x_min = int(df_plot['veh_saved'].min())
x_max = int(df_plot['veh_saved'].max())
axes[2, 1].set_xticks(range(x_min, x_max + 1))

axes[2, 1].set_title('Distribution of Fleet Vehicles Saved per Problem Instance', fontweight='bold', pad=10)
axes[2, 1].set_xlabel('Number of Trucks Saved (Greedy Fleet - OR-Tools Fleet)')
axes[2, 1].set_ylabel('Count of Instances')
axes[2, 1].legend()

# =========================================================================
# Save and Export
# =========================================================================
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 1. Save the complete 3x2 master dashboard
dashboard_filename = 'comprehensive_benchmark_dashboard.png'
fig.savefig(dashboard_filename, dpi=200)
print(f"Master 3x2 business analytics dashboard generated: {dashboard_filename}")

# 2. Automatically crop and save all 6 individual high-res charts
extent_00 = axes[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('1_incom2024_distance_comparison.png', bbox_inches=extent_00.expanded(1.2, 1.2), dpi=150)

extent_01 = axes[0, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('2_mim2025_distance_comparison.png', bbox_inches=extent_01.expanded(1.2, 1.2), dpi=150)

extent_10 = axes[1, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('3_improvement_distribution.png', bbox_inches=extent_10.expanded(1.2, 1.2), dpi=150)

extent_11 = axes[1, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('4_scale_vs_improvement.png', bbox_inches=extent_11.expanded(1.2, 1.2), dpi=150)

extent_20 = axes[2, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('5_distance_comparison_scatter.png', bbox_inches=extent_20.expanded(1.2, 1.2), dpi=150)

extent_21 = axes[2, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('6_vehicles_saved_distribution.png', bbox_inches=extent_21.expanded(1.2, 1.2), dpi=150)

print("All 6 individual high-res charts have been successfully extracted and saved!")
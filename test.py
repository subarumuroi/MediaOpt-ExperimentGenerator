from src.data.processing import load_and_filter_yarrowia_matrix
from src.analysis.descriptives import compute_compound_stats
from src.analysis.clustering import compute_ratio_distance_matrix, hierarchical_clustering, generate_merge_table
from src.generator.cluster_utils import generate_clusters, generate_lhs_samples, decode_bo_sample, build_sample_matrix
from src.generator.io_utils import export_samples, save_bo_setup, load_bo_setup
from src.generator.plotting import plot_sampled_ranges

import pandas as pd

file_path = ("data/Media-Matrix-Combined-grams-per-litre-Annotate.csv")

# Function usage
element_df, vitamin_df = load_and_filter_yarrowia_matrix(path=file_path)

# For Elements
drop_e = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3b_YAR','3c_YAR','6_YAR']
df_e_bounds = compute_compound_stats(element_df, drop_cols=drop_e)
std_matrix_e = compute_ratio_distance_matrix(element_df, drop_cols=drop_e)
Z_e_std = hierarchical_clustering(std_matrix_e, title="Element Clustering (Std)")
merge_df_e = generate_merge_table(Z_e_std, compounds=std_matrix_e.index.tolist())

# For Vitamins
drop_v = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3c_YAR','6_YAR']
df_v_bounds = compute_compound_stats(vitamin_df, drop_cols=drop_v)
std_matrix_v = compute_ratio_distance_matrix(vitamin_df, drop_cols=drop_v)
Z_v_std = hierarchical_clustering(std_matrix_v, title="Vitamin Clustering (Std)")
merge_df_v = generate_merge_table(Z_v_std, compounds=std_matrix_v.index.tolist())



# --- Generate clusters ---
clusters_meta_e = generate_clusters(merge_df_e, df_e_bounds, prefix="cluster_e", threshold=0.45)
clusters_meta_v = generate_clusters(merge_df_v, df_v_bounds, prefix="cluster_v", threshold=0.5)

# Combine element and vitamin bounds for LHS
combined_meta = {
    'named_clusters': {**clusters_meta_e['named_clusters'], **clusters_meta_v['named_clusters']},
    'bo_bounds': {**clusters_meta_e['bo_bounds'], **clusters_meta_v['bo_bounds']},
    'scaler_map': {**clusters_meta_e['scaler_map'], **clusters_meta_v['scaler_map']}
}

# --- Generate LHS samples ---
lhs_df = generate_lhs_samples(combined_meta, n_samples=191, random_state=42)


# Decode all LHS samples
decoded_samples = [
    decode_bo_sample(row.values, lhs_df.columns.tolist(), combined_meta['scaler_map'])
    for _, row in lhs_df.iterrows()
]

# --- Plot sampled vs expanded ranges using per-compound stats --
df_bounds_combined = pd.concat([df_e_bounds, df_v_bounds], ignore_index=True)
plot_sampled_ranges(decoded_samples, df_bounds_combined)

save_bo_setup(combined_meta)
              
# 3. Export samples for Julia
export_samples(decoded_samples, filename = 'initial_lhs_sample_conc.csv')
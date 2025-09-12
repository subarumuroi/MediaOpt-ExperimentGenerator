from src.data.processing import load_and_filter_yarrowia_matrix
from src.analysis.clustering import compute_ratio_distance_matrix, hierarchical_clustering, generate_merge_table

file_path = ("data/Media-Matrix-Combined-grams-per-litre-Annotate.csv")

# Function usage
element_df, vitamin_df = load_and_filter_yarrowia_matrix(path=file_path)

# For Elements
drop_e = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3b_YAR','3c_YAR','6_YAR']
std_matrix_e = compute_ratio_distance_matrix(element_df, drop_cols=drop_e)
Z_e_std = hierarchical_clustering(std_matrix_e, title="Element Clustering (Std)")
merge_df_e = generate_merge_table(Z_e_std, compounds=std_matrix_e.index.tolist())

# For Vitamins
drop_v = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3c_YAR','6_YAR']
std_matrix_v = compute_ratio_distance_matrix(vitamin_df, drop_cols=drop_v)
Z_v_std = hierarchical_clustering(std_matrix_v, title="Vitamin Clustering (Std)")
merge_df_v = generate_merge_table(Z_v_std, compounds=std_matrix_v.index.tolist())
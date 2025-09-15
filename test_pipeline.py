# test ideas from chatgpt, haven't tested yet.

import pytest
import numpy as np
import pandas as pd
import torch
from src.data.processing import load_and_filter_yarrowia_matrix
from src.analysis.clustering import compute_ratio_distance_matrix, hierarchical_clustering, generate_merge_table
from src.alpha_generator import descriptive_statistics, ClusterScaler, decode_bo_sample  # adjust import path

# --- Sample data path (small test dataset or subset of your real data) ---
FILE_PATH = "data/Media-Matrix-Combined-grams-per-litre-Annotate.csv"

# --- Fixtures for tests ---
@pytest.fixture
def element_vitamin_data():
    element_df, vitamin_df = load_and_filter_yarrowia_matrix(path=FILE_PATH)
    return element_df, vitamin_df

@pytest.fixture
def descriptive_stats(element_vitamin_data):
    element_df, vitamin_df = element_vitamin_data
    df_e_stats = descriptive_statistics(element_df)
    df_v_stats = descriptive_statistics(vitamin_df)
    return df_e_stats, df_v_stats

# --- Unit test: descriptive statistics ---
def test_descriptive_statistics_basic(descriptive_stats):
    df_e_stats, df_v_stats = descriptive_stats
    
    # Check required columns exist
    for df in [df_e_stats, df_v_stats]:
        for col in ["Compound", "Min", "Max", "Median", "Mean", "Expanded_Min", "Expanded_Max"]:
            assert col in df.columns, f"Column {col} missing"
            
    # Min <= Median <= Max
    for df in [df_e_stats, df_v_stats]:
        assert (df["Min"] <= df["Median"]).all()
        assert (df["Median"] <= df["Max"]).all()
        assert (df["Expanded_Min"] <= df["Expanded_Max"]).all()

# --- Unit test: ClusterScaler bounds ---
def test_cluster_scaler_bounds(descriptive_stats):
    df_e_stats, df_v_stats = descriptive_stats
    cluster = ["Copper (II) sulphate", "Zinc sulphate heptahydrate"]
    scaler = ClusterScaler(df_e_stats, cluster)
    alpha_min, alpha_max = scaler.get_alpha_bounds()
    
    # alpha_min should be <= alpha_max
    assert alpha_min <= alpha_max
    # concentrations computed from alpha should be within expanded bounds
    for compound in cluster:
        conc = scaler.concentration_from_alpha(alpha_max)[compound]
        max_val = df_e_stats.loc[df_e_stats["Compound"]==compound, "Expanded_Max"].values[0]
        assert conc <= max_val * 1.01  # allow tiny floating-point tolerance

# --- Unit test: LHS sampling + decoding ---
def test_lhs_decoding(descriptive_stats):
    df_e_stats, df_v_stats = descriptive_stats
    # small mock LHS sample for testing
    keys = ["cluster_test"]
    cluster = ["Copper (II) sulphate", "Zinc sulphate heptahydrate"]
    scaler = ClusterScaler(df_e_stats, cluster)
    scaler_map = {"cluster_test": scaler}
    
    sample = np.array([0.5])  # alpha
    decoded = decode_bo_sample(sample, keys, scaler_map, df_e_stats)
    
    # All compounds should be present
    for compound in cluster:
        assert compound in decoded
        # Should be within expanded bounds
        min_val = df_e_stats.loc[df_e_stats["Compound"]==compound, "Expanded_Min"].values[0]
        max_val = df_e_stats.loc[df_e_stats["Compound"]==compound, "Expanded_Max"].values[0]
        assert min_val <= decoded[compound] <= max_val * 1.01

# --- Integration test: full pipeline on a small subset ---
def test_full_pipeline(element_vitamin_data):
    element_df, vitamin_df = element_vitamin_data
    # Drop some columns to simulate standard preprocessing
    drop_e = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3b_YAR','3c_YAR','6_YAR']
    drop_v = ['1_YAR','2b_YAR','2c_YAR','2d_YAR','3c_YAR','6_YAR']
    
    # Distance matrix
    std_matrix_e = compute_ratio_distance_matrix(element_df, drop_cols=drop_e)
    std_matrix_v = compute_ratio_distance_matrix(vitamin_df, drop_cols=drop_v)
    
    # Check matrix is square
    assert std_matrix_e.shape[0] == std_matrix_e.shape[1]
    assert std_matrix_v.shape[0] == std_matrix_v.shape[1]
    
    # Clustering
    Z_e = hierarchical_clustering(std_matrix_e, title=None)
    Z_v = hierarchical_clustering(std_matrix_v, title=None)
    
    # Merge table
    merge_df_e = generate_merge_table(Z_e, compounds=std_matrix_e.index.tolist())
    merge_df_v = generate_merge_table(Z_v, compounds=std_matrix_v.index.tolist())
    
    # Check merge table columns exist
    for df in [merge_df_e, merge_df_v]:
        for col in ["Compound", "Cluster"]:
            assert col in df.columns

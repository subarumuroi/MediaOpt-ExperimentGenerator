from src.generator.cluster_scaler import ClusterScaler

import pandas as pd
import numpy as np
from pyDOE3 import lhs

from .cluster_scaler import ClusterScaler


def generate_clusters(
    merge_df,
    df_bounds,
    prefix="cluster",
    threshold=0.5,
    col1="Cluster_1",
    col2="Cluster_2",
):
    """
    Generate clusters of compounds based on linkage distance threshold,
    create ClusterScaler objects, compute ratios, and prepare BO bounds.

    Parameters
    ----------
    merge_df : pd.DataFrame
        Merge table from hierarchical clustering (must contain Cluster1, Cluster2, Distance).
    df_bounds : pd.DataFrame
        Bounds table with columns ['Compound', 'Min', 'Max', 'Median', ...].
    prefix : str
        Prefix for cluster naming.
    threshold : float
        Linkage distance threshold for merging clusters.
    col1, col2 : str
        Column names in merge_df specifying merged items.

    Returns
    -------
    dict
        {
            "named_clusters": {cluster_name: [compounds...]},
            "scaler_map": {cluster_name: ClusterScaler},
            "ratios": {cluster_name: {compound: ratio_to_base}},
            "bo_bounds": {cluster_name: (lo, hi), compound: (log_lo, log_hi), ...}
        }
    """
    def to_list(item):
        if isinstance(item, str):
            return [x.strip() for x in item.split(",")]
        elif isinstance(item, list):
            return item
        else:
            return [item]

    # --- Step 1: Build raw clusters from merge_df ---
    clusters = {}

    for _, row in merge_df.iterrows():
        if row["Linkage_Distance"] > threshold:
            continue

        c1 = to_list(row[col1])
        c2 = to_list(row[col2])

        # Merge existing clusters if any overlap
        for cname, comp_list in list(clusters.items()):
            if any(comp in comp_list for comp in c1):
                c1 = sorted(set(c1 + comp_list))
                del clusters[cname]
            if any(comp in comp_list for comp in c2):
                c2 = sorted(set(c2 + comp_list))
                del clusters[cname]

        merged = sorted(set(c1 + c2))
        # temp name, will be reassigned later
        clusters[f"temp{len(clusters)+1}"] = merged

    # --- Step 2: Reorder clusters systematically ---
    sorted_clusters = sorted(clusters.values(), key=lambda comps: comps)
    named_clusters = {
        f"{prefix}{i+1}": sorted(comps) for i, comps in enumerate(sorted_clusters)
    }

    # --- Step 3: Create ClusterScaler objects ---
    scaler_map = {
        cname: ClusterScaler(df_bounds, compounds)
        for cname, compounds in named_clusters.items()
    }

    # --- Step 4: Compute ratios using Median values ---
    ratios = {}
    for cname, compounds in named_clusters.items():
        sub_df = df_bounds[df_bounds["Compound"].isin(compounds)]
        if sub_df.empty:
            continue
        base = sorted(sub_df["Compound"])[0]  # alphabetical base
        base_val = sub_df[sub_df["Compound"] == base]["Median"].values[0]
        ratios[cname] = {
            comp: sub_df[sub_df["Compound"] == comp]["Median"].values[0] / base_val
            for comp in compounds
        }

    # --- Step 5: Collect non-clustered compounds ---
    clustered_compounds = sum(named_clusters.values(), [])
    non_cluster = df_bounds[~df_bounds["Compound"].isin(clustered_compounds)]

    non_cluster_bounds = {
        compound: (lo, hi)
        for compound, lo, hi in non_cluster[["Compound", "Expanded_Min", "Expanded_Max"]]
        .sort_values("Compound")  # alphabetical ordering
        .itertuples(index=False, name=None)
    }

    # --- Step 6: Get alpha bounds from scalers ---
    cluster_bounds = {
        cname: scaler_map[cname].get_alpha_bounds() for cname in named_clusters
    }

    # --- Step 7: Log-transform non-clustered bounds ---
    non_cluster_bounds_log = {
        comp: (np.log10(lo), np.log10(hi)) for comp, (lo, hi) in non_cluster_bounds.items()
    }

    # --- Step 8: Combine ---
    bo_bounds = {**cluster_bounds, **non_cluster_bounds_log}

    return {
        "named_clusters": named_clusters,
        "scaler_map": scaler_map,
        "ratios": ratios,
        "bo_bounds": bo_bounds,
    }

def generate_lhs_samples(clusters_meta, n_samples=191, random_state=42):
    """
    Generate Latin Hypercube Samples (LHS) for all clusters and non-clustered compounds.
    Combines elements and vitamins automatically, keeping the notebook workflow.

    Parameters
    ----------
    clusters_meta : dict
        Output of generate_clusters containing 'bo_bounds' key.
    n_samples : int
        Number of LHS samples to generate.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        LHS samples with compounds/clusters as columns and n_samples rows.
    """

    # Extract BO bounds
    bo_bounds = clusters_meta['bo_bounds']
    compounds = list(bo_bounds.keys())
    n_dims = len(compounds)

    # Lower and upper bounds
    lb = np.array([bo_bounds[c][0] for c in compounds])
    ub = np.array([bo_bounds[c][1] for c in compounds])

    # LHS in unit hypercube
    unit_lhs = lhs(n_dims, samples=n_samples, criterion='maximin', random_state=random_state)

    # Scale to bounds
    samples = lb + unit_lhs * (ub - lb)

    # Convert to DataFrame
    lhs_df = pd.DataFrame(samples, columns=compounds)

    return lhs_df
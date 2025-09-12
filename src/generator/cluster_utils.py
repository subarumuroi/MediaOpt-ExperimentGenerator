def generate_clusters(merge_df, df_bounds, prefix="cluster", threshold=0.5, col1=None, col2=None):
    """
    Automatically generate ClusterScaler objects from a merge table.
    
    Args:
        merge_df (pd.DataFrame): Output from generate_merge_table.
        df_bounds (pd.DataFrame): DataFrame with 'Compound', 'Min', 'Max', 'Median', etc.
        prefix (str): Cluster name prefix.
        threshold (float): Maximum linkage distance to merge clusters.
        col1, col2 (str, optional): Column names for cluster pairs. If None, autodetect.
    
    Returns:
        dict: {cluster_name: ClusterScaler}
    """
    # Auto-detect columns if not provided
    if col1 is None or col2 is None:
        possible_cols = merge_df.columns.str.lower()
        if 'cluster_1' in possible_cols and 'cluster_2' in possible_cols:
            col1, col2 = 'Cluster_1', 'Cluster_2'
        elif 'compound1' in possible_cols and 'compound2' in possible_cols:
            col1, col2 = 'compound1', 'compound2'
        else:
            raise ValueError("Could not detect appropriate cluster columns.")

    # Initialize clusters
    clusters = {name: [name] for name in merge_df[col1].tolist() + merge_df[col2].tolist()}

    # Merge clusters according to linkage distance
    for _, row in merge_df.iterrows():
        if row['Linkage_Distance'] > threshold:
            continue
        c1, c2 = row[col1], row[col2]
        merged = sorted(set(clusters[c1] + clusters[c2]))
        new_name = f"{prefix}{len(clusters) + 1}"
        clusters[new_name] = merged
        # Remove old entries
        del clusters[c1]
        del clusters[c2]

    # Create ClusterScaler objects
    scalers = {name: ClusterScaler(df_bounds, compounds) for name, compounds in clusters.items()}
    return scalers

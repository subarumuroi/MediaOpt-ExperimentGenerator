import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def compute_ratio_distance_matrix(df, eps=None, drop_cols=None):
    """
    Compute pairwise compound distance matrix based on the 
    standard deviation (std) of log10 ratios across samples.

    Args:
        df (pd.DataFrame): Table of compounds with sample columns. Must contain 'Compound' column.
        eps (float, optional): Small pseudocount to avoid log(0). Defaults to 1/1000th of smallest non-zero value.
        drop_cols (list, optional): Columns to drop from distance calculations.

    Returns:
        distance_matrix_std (pd.DataFrame): Pairwise std-based distance matrix
    """
    df = df.copy()
    sample_cols = df.columns.difference(['Compound', 'Annotate'])

    if drop_cols:
        sample_cols = sample_cols.difference(drop_cols)

    # Determine eps
    if eps is None:
        min_nonzero = df[sample_cols].replace(0, np.nan).min().min()
        eps = min_nonzero * 1e-3
        print(f"Using adaptive eps={eps}")
    else:
        print(f"Using user-defined eps={eps}")

    compound_map = df.set_index('Compound')[sample_cols]
    compounds = compound_map.index.tolist()

    # Initialize distance matrix
    distance_matrix_std = pd.DataFrame(0, index=compounds, columns=compounds, dtype=float)

    # Compute pairwise std distances
    for c1, c2 in itertools.combinations(compounds, 2):
        vals1 = compound_map.loc[c1]
        vals2 = compound_map.loc[c2]

        log_ratio_diff = (np.log10(vals1 + eps) - np.log10(vals2 + eps)).abs()
        std_dist = log_ratio_diff.replace(0, np.nan).std(skipna=True)

        if np.isnan(std_dist):
            std_dist = 0.0

        distance_matrix_std.loc[c1, c2] = distance_matrix_std.loc[c2, c1] = std_dist

    return distance_matrix_std

def hierarchical_clustering(distance_matrix, plot=True, title='Hierarchical Clustering'):
    '''
    Perform average-linkage clustering and optionally plot dendrogram.

    Args: 
        distance_matrix (pd.DataFrame): pairwise distance matrix
        plot (bool): whether to plot dendrogram
        title (str): title for the dendrogram plot

    Returns:
        Z (ndarray): linkage matrix from scipy.cluster.hierarchy.linkage
    '''
    
    condensed = squareform(distance_matrix.values)
    Z = linkage(condensed, method='average')

    if plot:
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=distance_matrix.index.tolist(), leaf_rotation=90)
        plt.yscale('symlog', linthresh=1e-2)
        plt.ylabel('Linkage Distance')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return Z

def generate_merge_table(Z, compounds, plot_elbow= True):
    """
    
    Generate merge table from hierarchical clustering linkage matrix.
    
    Args:
        Z (ndarray): linkage matrix from scipy.cluster.hierarchy.linkage
        compounds (list): original compound names (rows of distance matrix)
        plot_elbow (bool): whether to plot Linkage Distance vs Step
    
    Returns
        merge_df_sorted (pd.DataFrame): merge table sorted by Linkage Distance
    """

    n = len(compounds)
    merge_history = []
    clusters = {i: [compounds[i]] for i in range(n)}

    for step, (idx1, idx2, dist, sample_count) in enumerate(Z):
        idx1, idx2 = int(idx1), int(idx2)
        members1 = clusters[idx1]
        members2 = clusters[idx2]
        new_cluster = members1 + members2
        clusters[n + step] = new_cluster

        merge_history.append({
            'Step': step + 1,
            'Cluster_1': ', '.join(members1),
            'Cluster_2': ', '.join(members2),
            'New_Cluster': ', '.join(new_cluster),
            'Linkage_Distance': dist,
            'Num_Compounds': len(new_cluster)
        })

    merge_df = pd.DataFrame(merge_history)
    merge_df_sorted = merge_df.sort_values(by='Linkage_Distance', ascending=True)

    if plot_elbow:
        merge_df_sorted[['Step', 'Linkage_Distance']].plot(x='Step', y='Linkage_Distance', marker='o')
        plt.ylabel('Linkage Distance')
        plt.title('Elbow Plot for Clustering')
        plt.grid(True)
        plt.show()
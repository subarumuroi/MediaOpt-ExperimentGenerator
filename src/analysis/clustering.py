import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def compute_ratio_distance_matrix(df, eps=None, drop_cols=None):
    """
    This function allows for the principled reduction of chemical
    components to clusters based on std or max-min (components with 
    consistent ratios across samples are clustered together).Std is 
    standard, but max-min is more robust to outliers. Generally, for
    small datasets like here max-min should be preferred although, both
    methods led to the same results in the Yarrowia example

    Compute log values of components, std ratio distance, and max-min 
    ratio distance matrices from df (Element or Vitamin)
    
    Args: 
        df (pd.DataFrame): Compound table with sample columns
        eps (float, optional): small pseudocount to avoid log(0). If
            None, the smallest non-zero value in df is used to 
            caclculate 1/1000th of that value as eps.
        
    Returns:
        distance_matrix_std, distance_matrix_range (pd.DataFrame): 
        pariwise distances using std and min-max
    """

    df = df.copy()

    df = df.sample(frac=1) # shuffle row for randomness
    sample_cols = df.columns.difference(['Compound', 'Annotate'])

    if eps is None:
        min_nonzero = df[sample_cols].replace(0, np.nan).min().min()
        eps = min_nonzero*1e-3
        print(f"Using adaptive eps={eps}")
    else:
        print(f"Using user-defined eps={eps}")

    # --- log-ratio distances ---
    ratios = []
    compound_pairs = list(itertools.combinations(df['Compound'], 2))
    compound_map = df.set_index('Compound')[sample_cols]

    for c1, c2 in compound_pairs:
        vals1 = compound_map.loc[c1]
        vals2 = compound_map.loc[c2]
        ratio = (np.log10(vals1 + eps) - np.log10(vals2 + eps)).abs()
        if ratio.notna().sum() ==0:
            ratio[:] = eps 
        row = {'Compound_1': c1, 'Compound_2': c2, **ratio.to_dict()}
        ratios.append(row)

    ratio_df = pd.DataFrame(ratios)

    # Drop irrelevant columns if specified
    if drop_cols is not None:
        ratio_df = ratio_df.drop(columns=drop_cols, errors="ignore")
    # recompute sample cols after dropping
    sample_cols = ratio_df.columns.difference(['Compound_1', 'Compound_2'])

    # compute std per pair
    ratio_df['Std_Ratio'] = ratio_df[sample_cols].replace(0, np.nan).std(axis=1, skipna=True).fillna(0)

    # build distance matrix
    compounds = pd.unique(ratio_df[['Compound_1', 'Compound_2']].values.ravel())
    distance_matrix_std = pd.DataFrame(0, index=compounds, columns=compounds, dtype=float)
    for _, row in ratio_df.iterrows():
        c1, c2, std = row['Compound_1'], row['Compound_2'], row['Std_Ratio']
        distance_matrix_std.loc[c1, c2] = std
        distance_matrix_std.loc[c2, c1] = std
    np.fill_diagonal(distance_matrix_std.values, 0)

    # ---max-min range distance ---
    range_values = []
    for _, row in ratio_df. iterrows():
        vals = row[sample_cols]. replace(0, np.nan)
        rng = eps if vals.notna().sum() == 0 else max(vals.max() - vals.min(), eps)
        range_values.append(rng)
    ratio_df['Range_Ratio'] = range_values

    distance_matrix_range = pd.DataFrame(0, index=compounds, columns=compounds, dtype=float)
    for _, row in ratio_df.iterrows():
        c1, c2, rng = row['Compound_1'], row['Compound_2'], row['Range_Ratio']
        distance_matrix_range.loc[c1, c2] = rng
        distance_matrix_range.loc[c2, c1] = rng

    return distance_matrix_std, distance_matrix_range

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
        plt.title('Elbow Polot for Clustering')
        plt.grid(True)
        plt.show()
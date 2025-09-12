# src/data/cluster_scaler.py
import numpy as np

class ClusterScaler:
    def __init__(self, df_stats, compounds):
        """
        Args:
            df_stats (pd.DataFrame): Output of src/data/descriptives.py: compute_compound_stats
            compounds (list): Compounds in this cluster
        """
        self.compounds = compounds
        self.df = df_stats.set_index('Compound').loc[compounds]

        self.offsets = np.log10(self.df['Median']) - np.mean(np.log10(self.df['Median']))

    def concentration_from_alpha(self, alpha):
        return {c: 10**(alpha + offset) for c, offset in zip(self.compounds, self.offsets)}

    def solve_alpha_for_target(self, compound, target):
        if compound not in self.compounds:
            raise ValueError(f"{compound} not in cluster")
        idx = self.compounds.index(compound)
        return np.log10(target) - self.offsets[idx]

    def get_alpha_bounds(self):
        mins = np.log10(self.df['Expanded_Min']) - self.offsets
        maxs = np.log10(self.df['Expanded_Max']) - self.offsets
        alpha_min, alpha_max = mins.min(), maxs.max()
        if alpha_min > alpha_max:
            raise ValueError("Alpha bounds conflict")
        return alpha_min, alpha_max

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_sampled_ranges(decoded_samples, df_bounds, title="Sampled Concentrations vs Expanded Ranges", bo_suggestions=None):
    """
    Plot sampled concentrations vs original/expanded ranges with optional BO suggestions.
    
    Args:
        decoded_samples (list of dict): Each dict is compound → concentration (linear g/L)
        df_bounds (pd.DataFrame): Must contain columns:
            ['Compound', 'Min', 'Max', 'Median', 'Mean', 'Expanded_Min', 'Expanded_Max']
        title (str): Plot title
        bo_suggestions (list of dict or list of tuples (label, dict), optional):
            Each entry is either a dict of compound → conc, or a tuple with a label and dict.
    """
    # Collect sampled values per compound
    sampled = defaultdict(list)
    for sample in decoded_samples:
        for comp, val in sample.items():
            sampled[comp].append(val)

    # Merge sampled stats into df_bounds
    df = df_bounds.copy()
    df["Sampled_Min"] = df["Compound"].map(lambda c: np.min(sampled[c]) if c in sampled else np.nan)
    df["Sampled_Max"] = df["Compound"].map(lambda c: np.max(sampled[c]) if c in sampled else np.nan)
    df["Sampled_Median"] = df["Compound"].map(lambda c: np.median(sampled[c]) if c in sampled else np.nan)
    df["Sampled_Mean"] = df["Compound"].map(lambda c: np.mean(sampled[c]) if c in sampled else np.nan)

    # Sort for cleaner plotting
    df_sorted = df.sort_values("Median")

    # Plot
    plt.figure(figsize=(14, 6))
    for i, row in enumerate(df_sorted.itertuples()):
        y = i
        plt.plot([row.Min, row.Max], [y, y], color='black', lw=1.5)                 # original range
        plt.plot([row.Expanded_Min, row.Expanded_Max], [y, y], color='skyblue', lw=6, alpha=0.4)  # expanded
        plt.plot([row.Sampled_Min, row.Sampled_Max], [y, y], color='red', lw=3, alpha=0.5)         # sampled
        plt.plot(row.Sampled_Median, y, 'o', color='green')  # median
        plt.plot(row.Sampled_Mean, y, 'x', color='orange')   # mean

    # Overlay BO suggestions if provided
    if bo_suggestions:
        for j, entry in enumerate(bo_suggestions):
            if isinstance(entry, tuple):
                label, suggestion = entry
            else:
                label, suggestion = f"BO_{j+1}", entry
            for i, comp in enumerate(df_sorted['Compound']):
                if comp in suggestion:
                    plt.plot(suggestion[comp], i, 's', markersize=6, label=label if i == 0 else "")

    plt.yticks(range(len(df_sorted)), df_sorted['Compound'])
    plt.xlabel("Concentration (g/L)")
    plt.title(title)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if bo_suggestions:
        plt.legend(loc='lower right')
    plt.show()

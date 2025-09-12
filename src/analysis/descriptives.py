import pandas as pd
import numpy as np

def compute_compound_stats(df, drop_cols=None, expand_percent=0.1):
    """
    Compute descriptive statistics and expanded bounds for each compound.

    Args:
        df (pd.DataFrame): Compound dataframe. Must contain 'Compound'.
        drop_cols (list, optional): Sample columns to drop before computing stats.
        expand_percent (float, optional): Percent to expand min/max bounds.
            Example: 0.1 = expand by 10%. Defaults to 0.1.

    Returns:
        pd.DataFrame: Descriptive statistics with columns:
            Compound, Min, Max, Median, Mean, Expanded_Min, Expanded_Max
    """
    df_copy = df.copy()

    # Drop unwanted sample columns
    if drop_cols:
        df_copy = df_copy.drop(columns=drop_cols, errors="ignore")

    # Replace 0 with NaN for stats
    df_nan = df_copy.replace(0, np.nan)

    # Identify numeric sample columns
    numeric_cols = df_nan.select_dtypes(include=[np.number]).columns

    # Compute stats
    min_values = df_nan[numeric_cols].min(axis=1, skipna=True)
    max_values = df_nan[numeric_cols].max(axis=1, skipna=True)
    median_values = df_nan[numeric_cols].median(axis=1, skipna=True)
    mean_values = df_nan[numeric_cols].mean(axis=1, skipna=True)

    # Expanded bounds
    expanded_min = min_values / (1 + expand_percent)
    expanded_max = max_values * (1 + expand_percent)

    # Build descriptive table
    descriptives = pd.DataFrame({
        "Compound": df_nan["Compound"],
        "Min": min_values,
        "Max": max_values,
        "Median": median_values,
        "Mean": mean_values,
        "Expanded_Min": expanded_min,
        "Expanded_Max": expanded_max,
    })

    return descriptives

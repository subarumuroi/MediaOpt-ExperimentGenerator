import os
import pandas as pd
import json
import pickle

def save_bo_setup(combined_meta, save_dir="outputs/setup", prefix="bo_setup"):
    """
    Save named_clusters, bo_bounds (JSON) and scaler_map (Pickle) for BO.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Paths
    json_path = os.path.join(save_dir, f"{prefix}.json")
    pickle_path = os.path.join(save_dir, f"{prefix}_scalers.pkl")

    # Save JSON (named_clusters + bo_bounds)
    save_dict = {
        "named_clusters": combined_meta['named_clusters'],
        "bo_bounds": {k: (float(v[0]), float(v[1])) for k, v in combined_meta['bo_bounds'].items()},
    }
    with open(json_path, "w") as f:
        json.dump(save_dict, f, indent=4)
    
    # Save scalers (Pickle)
    with open(pickle_path, "wb") as f:
        pickle.dump(combined_meta['scaler_map'], f)
    
    print(f"BO setup saved at {json_path} and {pickle_path}")


def load_bo_setup(load_dir="outputs/setup", prefix="bo_setup"):
    """
    Load named_clusters, bo_bounds (JSON) and scaler_map (Pickle).
    """
    json_path = os.path.join(load_dir, f"{prefix}.json")
    pickle_path = os.path.join(load_dir, f"{prefix}_scalers.pkl")
    
    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    
    with open(pickle_path, "rb") as f:
        loaded_scalers = pickle.load(f)
    
    combined_meta = {
        "named_clusters": loaded_json["named_clusters"],
        "bo_bounds": loaded_json["bo_bounds"],
        "scaler_map": loaded_scalers
    }
    
    print(f"BO setup loaded from {json_path} and {pickle_path}")
    return combined_meta


def export_samples(decoded_samples, save_dir="outputs/samples", filename="sample_concentration.csv"):
    """
    Save decoded samples with compounds as rows and samples as columns.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(decoded_samples)
    
    # Transpose so compounds are rows
    df_t = df.T
    df_t.columns = [f"Sample_{i+1}" for i in range(df_t.shape[1])]
    df_t.insert(0, "Compound", df_t.index)

    # Optional: add units row
    units_row = pd.DataFrame([["g/L"] + [""] * (df_t.shape[1] - 1)], columns=df_t.columns)
    df_final = pd.concat([units_row, df_t], ignore_index=True)

    path = os.path.join(save_dir, filename)
    df_final.to_csv(path, index=False)
    print(f"Samples exported to {path}")
    return path


def load_samples(path):
    """
    Load samples exported by `export_samples_transposed`.
    Returns a DataFrame with compounds as rows and samples as columns.
    """
    df = pd.read_csv(path, skiprows=1)  # skip units row
    df.set_index("Compound", inplace=True)
    return df
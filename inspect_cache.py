import os
import yaml
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect raw data cache statistics")
    parser.add_argument("--config", default="gf/configs/config_clean_physics.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load Config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config.get("paths", {}).get("dataset_root")
    if not dataset_root:
        print("Error: dataset_root not found in config.")
        return

    print(f"Inspecting Cache Directory: {dataset_root}")
    
    val_index_path = os.path.join(dataset_root, "val_index.yaml")
    val_data_path = os.path.join(dataset_root, "val.pt")
    
    files_to_check = []
    
    if os.path.exists(val_index_path):
        print(f"Found index file: {val_index_path}")
        with open(val_index_path, 'r') as f:
            index = yaml.safe_load(f)
            if "shards" in index and len(index["shards"]) > 0:
                # Check first 3 shards
                for i in range(min(3, len(index["shards"]))):
                    fname = index["shards"][i]["file"]
                    files_to_check.append(os.path.join(dataset_root, fname))
    elif os.path.exists(val_data_path):
        print(f"Found single data file: {val_data_path}")
        files_to_check.append(val_data_path)
    else:
        print("Error: No validation data found (checked val_index.yaml and val.pt)")
        # Try train data if val not found
        train_path = os.path.join(dataset_root, "train.pt")
        if os.path.exists(train_path):
             print(f"Fallback to train data: {train_path}")
             files_to_check.append(train_path)
        else:
             return

    if not files_to_check:
        print("No files to check.")
        return

    print("\n" + "="*60)
    print(f"{'File':<30} | {'Shape':<15} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 60)

    for fpath in files_to_check:
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=True)
            # data is list of dicts usually
            if isinstance(data, list) and len(data) > 0:
                # Check first sample
                sample = data[0]
                if "g_obs" in sample:
                    g = sample["g_obs"]
                    print(f"{os.path.basename(fpath):<30} | {str(tuple(g.shape)):<15} | {g.min():.2e} | {g.max():.2e} | {g.mean():.2e} | {g.std():.2e}")
                    
                    # Heuristic check
                    if g.max() < 0:
                        print(f"  [WARNING] Data looks LOG-transformed (Max < 0)")
                    elif g.max() < 1e-4:
                        print(f"  [WARNING] Data looks extremely small (Max < 1e-4). Might be unnormalized linear or scaling issue.")
                    elif g.max() > 100:
                         print(f"  [INFO] Data looks like normal Linear scale.")
                else:
                    print(f"{os.path.basename(fpath):<30} | No 'g_obs' key")
            else:
                print(f"{os.path.basename(fpath):<30} | Empty or invalid format")
        except Exception as e:
            print(f"{os.path.basename(fpath):<30} | Error: {e}")

    print("="*60)
    print("\nConfig Settings:")
    print(f"  LDOS Log Enabled: {config.get('data', {}).get('ldos_transform', {}).get('log', {}).get('enabled')}")
    print(f"  LDOS Quantile Enabled: {config.get('data', {}).get('ldos_transform', {}).get('quantile', {}).get('enabled')}")

if __name__ == "__main__":
    main()

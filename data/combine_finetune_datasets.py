import numpy as np
from pathlib import Path

dataset_names = [
    "commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa",
    "logiqa", "reclor", "squad", "race", "xsum", "oasst1", "personachat",
    "proofwriter",
]

def combine_split(split_name):
    all_ids = []
    all_labels = []
    for name in dataset_names:
        ids = np.load(f"data/{name}/{split_name}_ids.npy")
        labels = np.load(f"data/{name}/{split_name}_labels.npy")
        all_ids.append(ids)
        all_labels.append(labels)
        print(f"{name}: {ids.shape}")

    combined_ids = np.concatenate(all_ids, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    print(f"Combined {split_name}: {combined_ids.shape}")

    # Shuffle both arrays using the SAME permutation, so each example's tokens
    # stay paired with its own mask (see -100.py / build_index.py teaching notes).
    rng = np.random.default_rng(seed=42)
    shuffle_order = rng.permutation(len(combined_ids))
    combined_ids = combined_ids[shuffle_order]
    combined_labels = combined_labels[shuffle_order]

    return combined_ids, combined_labels

output_dir = Path("data/combined")
output_dir.mkdir(parents=True, exist_ok=True)

train_ids, train_labels = combine_split("train")
np.save(output_dir / "train_ids.npy", train_ids)
np.save(output_dir / "train_labels.npy", train_labels)
print(f"Saved combined train set: {train_ids.shape}")

validation_ids, validation_labels = combine_split("validation")
np.save(output_dir / "validation_ids.npy", validation_ids)
np.save(output_dir / "validation_labels.npy", validation_labels)
print(f"Saved combined validation set: {validation_ids.shape}")

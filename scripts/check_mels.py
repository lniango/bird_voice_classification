import os
import torch

files = {
    "train": "data/mels/train.pt",
    "val": "data/mels/val.pt",
    "test": "data/mels/test.pt",
}

expected = {
    "train": 7858,
    "val": 1167,
    "test": 883,
}

for split, path in files.items():
    print(f"\n--- {split.upper()} ---")

    if not os.path.exists(path):
        print("ERROR: file does not exist")
        continue

    print(f"File size: {os.path.getsize(path) / (1024**2):.2f} MB")

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        features = data["features"]
        labels = data["labels"]

        print(f"Features shape: {tuple(features.shape)}")
        print(f"Labels shape:   {tuple(labels.shape)}")
        print(f"Feature dtype:  {features.dtype}")
        print(f"Label dtype:    {labels.dtype}")
        print(f"Expected examples: {expected[split]}")

        print(f"Number of features: {features.shape[0]}")
        print(f"Number of labels:   {labels.shape[0]}")
        print(f"NaN values: {torch.isnan(features).sum().item()}")
        print(f"Inf values: {torch.isinf(features).sum().item()}")
        print(f"Unique labels: {torch.unique(labels).tolist()}")

        if features.shape[0] != expected[split]:
            print("WARNING: unexpected number of examples!")

        if features.shape[0] != labels.shape[0]:
            print("WARNING: features and labels have different lengths!")

    except Exception as e:
        print(f"ERROR while reading file: {e}")

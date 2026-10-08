import pandas as pd
from pathlib import Path

df = pd.read_csv("annotations.csv")

# On retire les annotations inconnues
df = df[df["Species eBird Code"] != "????"].copy()

# Uniquement les fichiers audio réellement utilisés
files = sorted(df["Filename"].unique())

# Mélange reproductible
rng = __import__("random").Random(42)
rng.shuffle(files)

n = len(files)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_files = files[:n_train]
val_files = files[n_train:n_train + n_val]
test_files = files[n_train + n_val:]

splits = pd.DataFrame({
    "Filename": train_files + val_files + test_files,
    "split": (
        ["train"] * len(train_files)
        + ["val"] * len(val_files)
        + ["test"] * len(test_files)
    ),
})

splits.to_csv("splits.csv", index=False)

print(splits["split"].value_counts())
print()
print(splits.head(10))

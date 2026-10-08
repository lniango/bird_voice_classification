import pandas as pd

annotations = pd.read_csv("annotations.csv")
annotations = annotations[annotations["Species eBird Code"] != "????"]

splits = pd.read_csv("splits.csv")

df = annotations.merge(splits, on="Filename")

table = pd.crosstab(
    df["Species eBird Code"],
    df["split"]
)

print(table.to_string())

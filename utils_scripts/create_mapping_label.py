import pandas as pd

species = [
    "amepip",
    "clanut",
    "daejun",
    "gcrfin",
    "herthr",
    "rocwre",
    "whcspa",
]

mapping = pd.DataFrame({
    "Species eBird Code": species,
    "label": range(len(species)),
})

mapping.to_csv("labels.csv", index=False)

print(mapping.to_string(index=False))

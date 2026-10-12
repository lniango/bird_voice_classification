import sys
import torch
import matplotlib.pyplot as plt

sys.path.append("src")

from bird_classifier.model import BirdCNN

# Créer le CNN
model = BirdCNN(num_classes=7)

# Récupérer les poids de la première convolution
weights = model.features[0].weight.detach().cpu()

print("Forme des poids :", weights.shape)

# Afficher les 16 filtres
fig, axes = plt.subplots(4, 4, figsize=(10, 8))

for i, ax in enumerate(axes.flat):
    filt = weights[i, 0].numpy()

    im = ax.imshow(filt, cmap="viridis")
    ax.set_title(f"Filtre {i + 1}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Fréquence")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Filtres de la première convolution — avant entraînement")
plt.tight_layout()

plt.savefig("logs/initial_filters.png", dpi=150)
print("Image enregistrée dans logs/initial_filters.png")

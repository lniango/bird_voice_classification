import sys
import torch

sys.path.append("src")

from bird_classifier.dataset import BirdSoundDataset
from bird_classifier.model import BirdCNN


# Charger un véritable batch de spectrogrammes
dataset = BirdSoundDataset("data", split="train")
mel, label = dataset[0]

# Ajouter une dimension batch
x = mel.unsqueeze(0)

print("Entrée du CNN :", x.shape)
print("-" * 60)


# Créer le modèle
model = BirdCNN(num_classes=7)
model.eval()


# Afficher la sortie de chaque couche
with torch.no_grad():
    for name, layer in model.features.named_children():
        x = layer(x)
        print(f"{name:20s} -> {str(x.shape):25s} | {layer}")

    print("-" * 60)

    for name, layer in model.classifier.named_children():
        x = layer(x)
        print(f"{name:20s} -> {str(x.shape):25s} | {layer}")


print("-" * 60)
print("Sortie finale :", x)
print("Nombre de scores :", x.shape[1])

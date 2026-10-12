import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append("src")

from bird_classifier.dataset import PrecomputedMelDataset, AugmentedMelDataset
from bird_classifier.model import BirdCNN

import random
import numpy as np

train_num = 3

def set_seed(seed=42):
    """Random seed for reproductibility"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed) #generate random numbers for the current GPU
    torch.cuda.manual_seed_all(seed) # For all GPUs

    # Reprocductibility of cuda operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # use deterministic algorithms when available
    torch.use_deterministic_algorithms(True)


def main():
    # Set seed
    set_seed(seed=42)

    # Choisir le GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)

    # Charger les datasets
    train_dataset = AugmentedMelDataset("data/mels/train.pt")
    val_dataset = PrecomputedMelDataset("data/mels/val.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print(f"Exemples train : {len(train_dataset)}", flush=True)
    print(f"Exemples val   : {len(val_dataset)}", flush=True)

    # Créer le modèle, la fonction de perte et l'optimiseur
    model = BirdCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Boucle d'entraînement
    epochs = 20

    writer = SummaryWriter(log_dir=f"runs/bird_cnn_{train_num}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for mel, labels in train_loader:
            mel = mel.to(device)
            labels = labels.to(device)

            # Effacer les gradients précédents
            optimizer.zero_grad()

            # Prédictions
            outputs = model(mel)

            # Calcul de l'erreur
            loss = criterion(outputs, labels)

            # Calcul des gradients et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Statistiques d'entraînement
            train_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # 5. Évaluation sur la validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                labels = labels.to(device)

                outputs = model(mel)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/validation", val_loss, epoch + 1)

        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch + 1)

        writer.flush()  
      
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2%}",
            flush=True,
        )

    writer.close()

    # 6. Sauvegarder le modèle entraîné
    torch.save(
        model.state_dict(),
        f"checkpoints/bird_cnn{train_num}.pth",
    )
    print(f"Modèle sauvegardé dans checkpoints/bird_cnn{train_num}.pth", flush=True)


if __name__ == "__main__":
    main()

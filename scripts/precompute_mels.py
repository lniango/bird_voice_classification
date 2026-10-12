from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bird_classifier.dataset import BirdSoundDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "mels"
SPLITS = ("train", "val", "test")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dataset = BirdSoundDataset(DATA_DIR, split)
        output_file = OUTPUT_DIR / f"{split}.pt"

        if output_file.exists():
            print(f"{split}: fichier déjà présent, ignoré ({output_file})")
            continue

        features = []
        labels = []

        print(f"\nPré-calcul du split {split} ({len(dataset)} exemples)")

        for i in tqdm(range(len(dataset)), desc=split):
            mel, label = dataset[i]
            features.append(mel.squeeze(0).to(torch.float16))
            labels.append(label)

        torch.save(
            {
                "features": torch.stack(features),
                "labels": torch.stack(labels),
            },
            output_file,
        )

        print(f"Enregistré : {output_file}")
        print(f"Taille : {output_file.stat().st_size / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()

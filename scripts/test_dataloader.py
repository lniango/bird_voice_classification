import sys
sys.path.append("src")

from torch.utils.data import DataLoader
from bird_classifier.dataset import BirdSoundDataset


dataset = BirdSoundDataset(
    "data",
    split="train",
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

mel_batch, labels_batch = next(iter(loader))

print("Batch Mel :", mel_batch.shape)
print("Batch labels :", labels_batch.shape)
print("Type Mel :", mel_batch.dtype)
print("Labels présents :", labels_batch.unique())

import sys
sys.path.append("src")

from bird_classifier.dataset import BirdSoundDataset

dataset = BirdSoundDataset(
    "data",
    split="train",
)

mel, label = dataset[0]

print("Mel-spectrogram :", mel.shape)
print("Label           :", label)
print("Type             :", mel.dtype)

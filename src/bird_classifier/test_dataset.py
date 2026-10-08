from dataset import * 

dataset = BirdSoundDataset(
    "../../data",
    split="train",
)

print("Nombre d'exemples :", len(dataset))

waveform, label = dataset[0]

print("Waveform :", waveform.shape)
print("Label    :", label)

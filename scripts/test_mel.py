import sys
sys.path.append("src")

import matplotlib.pyplot as plt
import torchaudio

from bird_classifier.dataset import BirdSoundDataset


dataset = BirdSoundDataset(
    "data",
    split="train",
)

waveform, label = dataset[0]

print("Waveform :", waveform.shape)
print("Label    :", label)


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
)

mel = mel_transform(waveform)

db_transform = torchaudio.transforms.AmplitudeToDB()
mel_db = db_transform(mel)

print("Mel-spectrogram :", mel.shape)
print("Mel dB range :", mel_db.min().item(), "to", mel_db.max().item())

plt.figure(figsize=(10, 4))

plt.imshow(
    mel_db[0].numpy(),
    origin="lower",
    aspect="auto",
)

plt.xlabel("Time")
plt.ylabel("Mel frequency")
plt.title(f"Mel-spectrogram — class {label}")

plt.colorbar(label="Energy")

plt.tight_layout()

#save spectogram
plt.savefig(
    "logs/mel_spectrogram.png",
    dpi=150,
    bbox_inches="tight",
)

print("Image sauvegardée dans logs/mel_spectrogram.png")

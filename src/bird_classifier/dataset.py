from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class BirdSoundDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)

        annotations = pd.read_csv(
                self.data_dir / "annotations.csv"
                )

        splits = pd.read_csv(
                self.data_dir / "splits.csv"
                )

        labels = pd.read_csv(
                self.data_dir / "labels.csv"
                )

        # Keep only the 7 selected species
        annotations = annotations[
                annotations["Species eBird Code"].isin(
                    labels["Species eBird Code"]
                    )
                ].copy()

        # Add Splitting train / val / test
        annotations = annotations.merge(
                splits,
                on="Filename",
                how="inner",
                )

        # Keep only required split
        self.annotations = annotations[
                annotations["split"] == split
                ].reset_index(drop=True)

        #Mapping species
        self.label_map = dict(
                zip(
                    labels["Species eBird Code"],
                    labels["label"],
                    )
                )

        self.sample_rate = 32000
        self.target_duration = 1.0
        self.target_frames = int(
                self.sample_rate * self.target_duration
                )
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate = self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64,
                )




    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        
        audio_path = self.data_dir / "audio" / row["Filename"]

        start_time = float(row["Start Time (s)"])
        end_time   = float(row["End Time (s)"])

        frame_offset = int(start_time * self.sample_rate)
        # Take at least 1 frame
        num_frames = max(
                1,
                int(
                (end_time - start_time) * self.sample_rate
                )
                )

        waveform, sample_rate = torchaudio.load(
                audio_path,
                frame_offset = frame_offset,
                num_frames = num_frames,
                )

        #Extract 1sec 
        if waveform.shape[1] < self.target_frames:
            padding = self.target_frames - waveform.shape[1]

            waveform = torch.nn.functional.pad(
                    waveform,
                    (0, padding),
                    )

        elif waveform.shape[1] > self.target_frames:
            waveform = waveform[:, :self.target_frames]

        #Mel spectogram
        mel = self.mel_transform(waveform)

        label = self.label_map[row["Species eBird Code"]]

        return mel, torch.tensor(label, dtype=torch.long)





        

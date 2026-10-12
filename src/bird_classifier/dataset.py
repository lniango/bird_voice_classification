from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from functools import lru_cache
from torchaudio.transforms import FrequencyMasking, TimeMasking


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

    @lru_cache(maxsize=16)
    def _load_audio(self, filename):
        audio_path = self.data_dir / "audio" / filename
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.sample_rate:
            raise ValueError(
                    f"Sample rate inattendu pour {filename}: "
                    f"{sample_rate} Hz, attendu {self.sample_rate} Hz"
                    )
        return waveform


    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        
        #audio_path = self.data_dir / "audio" / row["Filename"]
        audio = self._load_audio(row["Filename"])

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

        """waveform, sample_rate = torchaudio.load(
                audio_path,
                frame_offset = frame_offset,
                num_frames = num_frames,
                )"""
        waveform = audio[:, frame_offset:frame_offset + num_frames]

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


class PrecomputedMelDataset(Dataset):
    """Load mel precomputed spectograms from .pt files"""
    def __init__(self, file_path):
        data = torch.load(
                file_path,
                map_location="cpu",
                weights_only=True
                )

        self.features = data["features"]
        self.labels = data["labels"]

        assert len(self.features) == len(self.labels), (
                "The nb of spectograms must matching nb of labels")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mel = self.features[index].unsqueeze(0).float()
        label = self.labels[index]

        return mel, label


class AugmentedMelDataset(Dataset):
    """Precomputed Mel spectograms with SpecAugment for training."""

    def __init__(
        self,
        file_path,
        freq_mask_param=8,
        time_mask_param=8,
        num_mask=2,
        ):
        
        data = torch.load(file_path, map_location='cpu', weights_only=True)

        self.features = data["features"]
        self.labels = data["labels"]

        self.freq_masking = FrequencyMasking(
                    freq_mask_param=freq_mask_param,
                    )
        self.time_masking = TimeMasking(
                time_mask_param=time_mask_param)
        

        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # initial shape [64, 63)
        mel = self.features[index].float()
        # Add channel (1, 64, 63)
        mel = mel.unsqueeze(0)   

        # Apply augmentations
        mel = self.freq_masking(mel)
        mel = self.time_masking(mel)

        label = self.labels[index]

        return mel, label


                    

        

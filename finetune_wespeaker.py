import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pyannote.core import Annotation, Segment

from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34


# ─── 1. Dataset ──────────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """Produces fixed-length waveform crops labelled with a global speaker ID."""

    def __init__(self, wav_scp, rttm_file, duration=3.0, sample_rate=16000, augment=False, speaker2idx=None):
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.augment = augment

        # Load audio paths
        self.audio_paths = {}
        with open(wav_scp) as f:
            for line in f:
                uid, path = line.strip().split(maxsplit=1)
                self.audio_paths[uid] = path

        # Build per-speaker segment list (global speaker IDs)
        self.speaker2idx = speaker2idx if speaker2idx is not None else {}
        self.samples = []  # list of (audio_path, start_s, end_s, spk_idx)
        self.fixed_vocab = speaker2idx is not None

        with open(rttm_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                uid = parts[1]
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                end = start + dur
                if uid not in self.audio_paths:
                    continue
                if spk not in self.speaker2idx:
                    if self.fixed_vocab:
                        continue
                    else:
                        self.speaker2idx[spk] = len(self.speaker2idx)
                spk_idx = self.speaker2idx[spk]
                # Only keep segments long enough to crop from
                if dur >= self.duration:
                    self.samples.append((self.audio_paths[uid], start, end, spk_idx))

        print(f"[Dataset] {len(self.samples)} usable segments, {len(self.speaker2idx)} unique speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, end, spk_idx = self.samples[idx]

        # Random crop within the labelled segment
        max_start = end - self.duration
        crop_start = random.uniform(start, max_start) if max_start > start else start
        crop_end = crop_start + self.duration

        waveform, sr = torchaudio.load(
            path,
            frame_offset=int(crop_start * sr if False else crop_start * self.sample_rate),
            num_frames=self.num_samples,
        )
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Ensure mono and correct length
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)
        if waveform.shape[-1] < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., :self.num_samples]

        # Waveform augmentation to prevent overfitting
        if self.augment:
            # 1. Random gain perturbation (0.5 to 1.5)
            if random.random() < 0.8:
                waveform = waveform * random.uniform(0.5, 1.5)
            
            # 2. Additive Gaussian noise (SNR 15dB to 30dB)
            if random.random() < 0.5:
                noise = torch.randn_like(waveform)
                clean_power = waveform.pow(2).mean()
                noise_power = noise.pow(2).mean()
                if noise_power > 0 and clean_power > 0:
                    snr = random.uniform(15.0, 30.0)
                    factor = (clean_power / (10 ** (snr / 10.0) * noise_power)).sqrt()
                    waveform = waveform + factor * noise

        return waveform, spk_idx  # (1, T), int


# ─── 2. LightningModule Wrapper ───────────────────────────────────────────────

class WeSpeakerFinetuner(pl.LightningModule):
    def __init__(self, num_classes, embedding_dim=256, lr=1e-4, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.backbone = WeSpeakerResNet34()
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # ArcFace-style classification head (AAM-Softmax)
        import pytorch_metric_learning.losses as pml_losses
        self.loss_func = pml_losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=28.6,
            scale=64.0,
        )

    def forward(self, waveforms):
        # waveforms: (B, 1, T)  → embeddings: (B, D)
        return self.backbone(waveforms)

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        embeddings = self(waveforms)
        loss = self.loss_func(embeddings, labels)
        
        # Calculate classification accuracy
        with torch.no_grad():
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            w_norm = F.normalize(self.loss_func.W, p=2, dim=0)
            cos_sim = torch.matmul(embeddings_norm, w_norm)
            preds = torch.argmax(cos_sim, dim=1)
            acc = (preds == labels).float().mean()
            
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch
        embeddings = self(waveforms)
        loss = self.loss_func(embeddings, labels)
        
        # Calculate classification accuracy
        with torch.no_grad():
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            w_norm = F.normalize(self.loss_func.W, p=2, dim=0)
            cos_sim = torch.matmul(embeddings_norm, w_norm)
            preds = torch.argmax(cos_sim, dim=1)
            acc = (preds == labels).float().mean()
            
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # WeSpeaker backbone uses a 100x smaller learning rate to preserve pretrained representations,
        # and we use AdamW with weight decay to prevent overfitting.
        return torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr": self.hparams.lr * 0.01},
            {"params": self.loss_func.parameters(), "lr": self.hparams.lr}
        ], weight_decay=1e-4)



# ─── 3. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = "/home3/yihao/Research/Code/DiariZen/recipes/diar_ssl/data/custom_finetune"
    MODEL_PATH = "/home3/yihao/Research/Code/DiariZen/recipes/base_model/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
    OUT_DIR = "exp_wespeaker_finetune"
    DURATION = 3.0
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 100
    LR = 1e-3

    # Build datasets
    train_ds = SpeakerDataset(f"{BASE}/train/wav.scp", f"{BASE}/train/rttm",
                               duration=DURATION, augment=True)
    val_ds   = SpeakerDataset(f"{BASE}/val/wav.scp",   f"{BASE}/val/rttm",
                               duration=DURATION, augment=False, speaker2idx=train_ds.speaker2idx)

    num_classes = len(train_ds.speaker2idx)
    print(f"Total speakers: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Initialize model
    model = WeSpeakerFinetuner(
        num_classes=num_classes,
        embedding_dim=256,
        lr=LR,
        pretrained_path=MODEL_PATH,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(OUT_DIR, "checkpoints"),
        filename="wespeaker-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(monitor="val/loss", patience=15, mode="min")

    # Trainer
    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=OUT_DIR,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print("Fine-tuning complete!")

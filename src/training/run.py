"""Training entry points for CTC and LLM experiments.

Shared by CLI (scripts/train.py) and Modal (modal_app.py).
"""

import logging

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.collate import ctc_collate_fn, simple_audio_collate_fn
from src.data.datasets import MLSDataset
from src.models.encoder import EncoderWrapper

log = logging.getLogger(__name__)


def build_dataloaders(cfg: DictConfig, collate_fn) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""
    train_ds = MLSDataset(
        split=cfg.data.train_split,
        max_duration_s=cfg.data.max_duration_s,
    )
    val_ds = MLSDataset(
        split=cfg.data.val_split,
        max_duration_s=cfg.data.max_duration_s,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, val_loader


def train_ctc(cfg: DictConfig, device: str):
    """Set up and run CTC training."""
    from src.models.ctc_model import CTCModel
    from src.training.ctc_trainer import CTCTrainer

    encoder = EncoderWrapper(cfg.encoder.name)
    model = CTCModel(encoder)
    train_loader, val_loader = build_dataloaders(cfg, ctc_collate_fn)

    trainer = CTCTrainer(model, train_loader, val_loader, cfg, device)
    trainer.train()


def train_llm(cfg: DictConfig, device: str):
    """Set up and run LLM adapter training."""
    from src.models.adapters import build_adapter
    from src.models.llm_model import LLMModel
    from src.training.llm_trainer import LLMTrainer

    encoder = EncoderWrapper(cfg.encoder.name)
    adapter = build_adapter(cfg.adapter)
    model = LLMModel(
        encoder=encoder,
        adapter=adapter,
        llm_name=cfg.llm.name,
        llm_dtype=getattr(torch, cfg.llm.dtype),
        prompt=cfg.llm.prompt,
    )
    train_loader, val_loader = build_dataloaders(cfg, simple_audio_collate_fn)

    trainer = LLMTrainer(model, train_loader, val_loader, cfg, device)
    trainer.train()

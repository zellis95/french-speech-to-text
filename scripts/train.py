"""Hydra CLI entry point for training. Dispatches to CTC or LLM trainer."""

import logging

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.training.run import train_ctc, train_llm
from src.utils.device import get_device

log = logging.getLogger(__name__)


def _resolve_device(cfg: DictConfig) -> str:
    if cfg.device == "auto":
        return get_device()
    return cfg.device


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    load_dotenv()

    device = _resolve_device(cfg)
    log.info(f"Device: {device}")
    log.info(f"Experiment: {cfg.experiment.name} ({cfg.experiment.type})")

    torch.manual_seed(cfg.seed)

    if cfg.experiment.type == "ctc":
        train_ctc(cfg, device)
    elif cfg.experiment.type == "llm":
        train_llm(cfg, device)
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment.type}")


if __name__ == "__main__":
    main()

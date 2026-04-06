"""Modal POC: run CTC training on a T4 GPU.

Usage:
    modal run modal_app.py
    modal run modal_app.py --overrides "training.epochs=5,data.max_duration_s=15.0"
    modal run modal_app.py --overrides "experiment=llm_adapter,adapter=conv_mlp"
"""

import modal

app = modal.App("french-asr-poc")

ckpt_vol = modal.Volume.from_name("asr-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")  # system dep for torchcodec/torchaudio
    .uv_sync()  # installs third-party deps from pyproject.toml + uv.lock
    .add_local_python_source("src")  # our code — goes to /root/src, on PYTHONPATH
    .add_local_dir("conf", remote_path="/root/conf")  # Hydra configs
)

# Defaults for Modal runs — always applied, then user overrides on top
_BASE_OVERRIDES = [
    "mode=local",  # uses dev split (~158MB), not full train (~17GB)
]


@app.function(
    image=image,
    gpu="T4",
    memory=(16384, 32768),  # 16GB request, 32GB hard limit (OOM-kills if exceeded)
    timeout=1800,  # 30 min — plenty for POC
    volumes={"/root/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_dotenv()],
)
def train(overrides: str = ""):
    """Run training on Modal GPU. Accepts comma-separated Hydra overrides."""
    import logging

    import torch
    from hydra import compose, initialize_config_dir

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Build override list: base defaults + user overrides
    override_list = list(_BASE_OVERRIDES)
    if overrides:
        override_list.extend(overrides.split(","))

    with initialize_config_dir(config_dir="/root/conf", version_base=None):
        cfg = compose(config_name="config", overrides=override_list)

    log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    log.info(f"Experiment: {cfg.experiment.name}")
    log.info(f"Overrides: {override_list}")

    torch.manual_seed(cfg.seed)

    from src.training.run import train_ctc, train_llm

    if cfg.experiment.type == "ctc":
        train_ctc(cfg, "cuda")
    elif cfg.experiment.type == "llm":
        train_llm(cfg, "cuda")
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment.type}")

    # Persist checkpoints to volume
    ckpt_vol.commit()
    log.info("Checkpoints committed to volume")


@app.local_entrypoint()
def main(overrides: str = ""):
    train.remote(overrides=overrides)

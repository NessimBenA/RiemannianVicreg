import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from lanistr_impl.data import setup_data
from lanistr_impl.models.lanistr_model import build_lanistr_model
from lanistr_impl.trainer import LANISTRTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_unique_output_dir(base: str, seed: int) -> str:
    base_path = Path(base)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    unique_path = base_path / f"{stamp}_pid{os.getpid()}_seed{seed}"
    unique_path.mkdir(parents=True, exist_ok=False)
    logger.info(f"Created unique output directory: {unique_path}")
    return str(unique_path)


def load_config(config_path: str, overrides: list[str]) -> DictConfig:
    base_config = OmegaConf.load(config_path)

    if "defaults" in base_config:
        defaults = base_config.pop("defaults")
        config_dir = Path(config_path).parent
        for default in defaults:
            if isinstance(default, str):
                default_path = config_dir / f"{default}.yaml"
                if default_path.exists():
                    default_config = OmegaConf.load(default_path)
                    base_config = OmegaConf.merge(default_config, base_config)

    if overrides:
        cli_config = OmegaConf.from_cli(overrides)
        base_config = OmegaConf.merge(base_config, cli_config)

    return base_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LANISTR multimodal model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config values (e.g., training.batch_size=64)",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))

    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.device = "cpu"

    logger.info("Loading data...")
    dataloaders, tabular_info, tokenizer = setup_data(config)

    num_classes = tabular_info["num_classes"]
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Tabular input dimension: {tabular_info['input_dim']}")

    logger.info("Building model...")
    model = build_lanistr_model(
        config=config,
        tabular_data_info=tabular_info,
        num_classes=num_classes,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if config.training.get("output_dir"):
        config.training.output_dir = make_unique_output_dir(config.training.output_dir, config.seed)
    else:
        config.output_dir = make_unique_output_dir(config.output_dir, config.seed)

    trainer = LANISTRTrainer(
        model=model,
        config=config,
        dataloaders=dataloaders,
    )

    training_mode = config.training.mode
    logger.info(f"Starting training in mode: {training_mode}")

    if training_mode == "supervised":
        trainer.train_supervised()
    elif training_mode == "simsiam":
        trainer.train_simsiam()
    elif training_mode == "simsiam_finetune":
        trainer.train_simsiam_then_finetune(tabular_info, num_classes)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

    if config.training.get("do_test", True):
        logger.info("Running test evaluation...")
        trainer.test()

    record = {
        "dataset_type": str(config.dataset.dataset_type),
        "ssl_objective": str(config.training.ssl_objective),
        "seed": int(config.seed),
        "image": bool(config.dataset.image),
        "text": bool(config.dataset.text),
        "tab": bool(config.dataset.tab),
        "summary": trainer.summary,
    }
    metrics_path = trainer.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    logger.info(f"Wrote {metrics_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()

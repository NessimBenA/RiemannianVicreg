from typing import Any, Callable, Literal, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.mug import load_mug
from data_loaders.petfinder import load_petfinder
from data_loaders.dataset_utils import ImageMaskGenerator


class LANISTRCollator:
    def __init__(
        self,
        mode: Literal["pretrain", "finetune"],
        image_mask_generator: Optional[ImageMaskGenerator] = None,
    ) -> None:
        self.mode = mode
        self.image_mask_generator = image_mask_generator

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        collated: dict[str, torch.Tensor] = {}

        if "pixel_values" in batch[0]:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            collated["pixel_values"] = pixel_values

        if "input_ids" in batch[0]:
            input_ids = torch.stack([item["input_ids"] for item in batch])
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(1)
            collated["input_ids"] = input_ids

            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
            collated["attention_mask"] = attention_mask

        if "features" in batch[0]:
            collated["features"] = torch.stack([item["features"] for item in batch])

        if "labels" in batch[0]:
            collated["labels"] = torch.tensor([item["labels"] for item in batch])

        if self.mode == "pretrain":
            if self.image_mask_generator is not None and "pixel_values" in collated:
                bool_masked_pos = torch.stack(
                    [self.image_mask_generator() for _ in batch]
                )
                collated["bool_masked_pos"] = bool_masked_pos

        return collated


def adapt_config_for_dataset_loader(config: DictConfig) -> DictConfig:
    adapted = OmegaConf.to_container(config, resolve=True)

    adapted["data_dir"] = config.dataset.data_dir
    adapted["text"] = config.dataset.text
    adapted["image"] = config.dataset.image
    adapted["tab"] = config.dataset.tab
    adapted["task"] = config.training.task
    adapted["max_token_length"] = config.encoders.text.max_token_length
    adapted["image_size"] = config.encoders.image.image_size
    adapted["image_encoder_name"] = config.encoders.image.name

    if hasattr(config.dataset, "dataset_type"):
        adapted["dataset_type"] = config.dataset.dataset_type

    if hasattr(config.dataset, "test_ratio"):
        adapted["test_ratio"] = config.dataset.test_ratio
    if hasattr(config.dataset, "val_ratio"):
        adapted["val_ratio"] = config.dataset.val_ratio

    return OmegaConf.create(adapted)


def get_dataset(
    config: DictConfig,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    dataset_name = config.dataset.name.lower()
    adapted_config = adapt_config_for_dataset_loader(config)

    if dataset_name == "mug":
        return load_mug(adapted_config, tokenizer)
    elif dataset_name == "petfinder":
        return load_petfinder(adapted_config, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_collator(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    tabular_input_dim: int,
) -> LANISTRCollator:
    task = config.training.task

    image_mask_gen: Optional[ImageMaskGenerator] = None

    if task == "pretrain":
        if config.dataset.image:
            image_mask_gen = ImageMaskGenerator(
                input_size=config.encoders.image.image_size,
                mask_patch_size=config.encoders.image.mask_patch_size,
                model_patch_size=config.encoders.image.patch_size,
                mask_ratio=config.encoders.image.masking_ratio,
            )

    return LANISTRCollator(
        mode=task,
        image_mask_generator=image_mask_gen,
    )


def get_dataloaders(
    config: DictConfig,
    dataset: dict[str, Any],
    collate_fn: Callable,
) -> dict[str, DataLoader]:
    workers = config.training.get("workers", 4)
    train_batch_size = config.training.batch_size
    eval_batch_size = config.training.get("eval_batch_size", train_batch_size)

    dataloaders: dict[str, DataLoader] = {}

    if "train" in dataset:
        dataloaders["train"] = DataLoader(
            dataset["train"],
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    if "valid" in dataset:
        dataloaders["valid"] = DataLoader(
            dataset["valid"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    if "test" in dataset:
        dataloaders["test"] = DataLoader(
            dataset["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return dataloaders


def setup_data(
    config: DictConfig,
) -> tuple[dict[str, DataLoader], dict[str, Any], AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.encoders.text.name)

    dataset = get_dataset(config, tokenizer)
    tabular_info = dataset["tabular_data_information"]

    collator = get_collator(config, tokenizer, tabular_info["input_dim"])
    dataloaders = get_dataloaders(config, dataset, collator)

    return dataloaders, tabular_info, tokenizer

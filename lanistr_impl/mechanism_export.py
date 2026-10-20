from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .models.lanistr_model import LANISTRFinetuneModel, LANISTRPretrainModel

MASKED_VIEWS: Final[int] = 3


class TransformDataset(Protocol):
    transform: object

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Tensor]: ...


def _reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_for_split(
    dataloaders: dict[str, DataLoader[dict[str, Tensor]]],
    split: str,
    batch_size: int,
) -> tuple[DataLoader[dict[str, Tensor]], object]:
    source = dataloaders[split]
    dataset = cast(TransformDataset, source.dataset)
    test_dataset = cast(TransformDataset, dataloaders["test"].dataset)
    original_transform = dataset.transform
    dataset.transform = test_dataset.transform
    loader = DataLoader(
        cast(Dataset[Mapping[str, Tensor]], dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=source.collate_fn,
        drop_last=False,
    )
    return loader, original_transform


def _restore_transform(
    dataloaders: dict[str, DataLoader[dict[str, Tensor]]],
    split: str,
    transform: object,
) -> None:
    dataset = cast(TransformDataset, dataloaders[split].dataset)
    dataset.transform = transform


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _concatenate(parts: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _pretrain_pass(
    model: LANISTRPretrainModel,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    seed: int,
) -> dict[str, NDArray[np.float32] | NDArray[np.int64]]:
    _reset_seed(seed)
    model.eval()
    model.capture_representations = True
    fusion_unmasked: list[NDArray[np.float32]] = []
    fusion_masked: list[NDArray[np.float32]] = []
    projector_unmasked: list[NDArray[np.float32]] = []
    projector_masked: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = _move_batch(batch, device)
            model(device_batch)
            capture = model.representation_capture
            if capture is None:
                raise RuntimeError("Representation capture was not populated")
            fusion_unmasked.append(capture.fusion_unmasked.cpu().numpy().astype(np.float32))
            fusion_masked.append(capture.fusion_masked.cpu().numpy().astype(np.float32))
            projector_unmasked.append(capture.projector_unmasked.cpu().numpy().astype(np.float32))
            projector_masked.append(capture.projector_masked.cpu().numpy().astype(np.float32))
            labels.append(device_batch["labels"].cpu().numpy().astype(np.int64))
    model.capture_representations = False
    model.representation_capture = None
    return {
        "fusion_unmasked": _concatenate(fusion_unmasked),
        "fusion_masked": _concatenate(fusion_masked),
        "projector_unmasked": _concatenate(projector_unmasked),
        "projector_masked": _concatenate(projector_masked),
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
    }


def export_pretrain_representations(
    model: LANISTRPretrainModel,
    config: DictConfig,
    dataloaders: dict[str, DataLoader[dict[str, Tensor]]],
    output_dir: Path,
) -> None:
    device = model.device
    batch_size = int(config.training.get("eval_batch_size", config.training.batch_size))
    for split_index, split in enumerate(("train", "test")):
        loader, original_transform = _loader_for_split(dataloaders, split, batch_size)
        first = _pretrain_pass(model, loader, device, int(config.seed) * 1000 + split_index * 100)
        masked_fusion = [cast(NDArray[np.float32], first["fusion_masked"])]
        masked_projector = [cast(NDArray[np.float32], first["projector_masked"])]
        unmasked_fusion = cast(NDArray[np.float32], first["fusion_unmasked"])
        unmasked_projector = cast(NDArray[np.float32], first["projector_unmasked"])
        for view_index in range(1, MASKED_VIEWS):
            result = _pretrain_pass(
                model,
                loader,
                device,
                int(config.seed) * 1000 + split_index * 100 + view_index,
            )
            next_fusion = cast(NDArray[np.float32], result["fusion_unmasked"])
            next_projector = cast(NDArray[np.float32], result["projector_unmasked"])
            if not np.allclose(unmasked_fusion, next_fusion, rtol=0.0, atol=1e-6):
                raise RuntimeError("Unmasked fusion representations changed across mask views")
            if not np.allclose(unmasked_projector, next_projector, rtol=0.0, atol=1e-6):
                raise RuntimeError("Unmasked projector representations changed across mask views")
            masked_fusion.append(cast(NDArray[np.float32], result["fusion_masked"]))
            masked_projector.append(cast(NDArray[np.float32], result["projector_masked"]))
        _restore_transform(dataloaders, split, original_transform)
        if model.ssl_objective == "rvicreg":
            kappas = model.get_kappas_product().detach().cpu().numpy().astype(np.float32)
        else:
            kappas = np.zeros(1, dtype=np.float32)
        labels = cast(NDArray[np.int64], first["labels"])
        np.savez_compressed(
            output_dir / f"pretrain_{split}.npz",
            indices=np.arange(labels.shape[0], dtype=np.int64),
            labels=labels,
            fusion_unmasked=unmasked_fusion,
            fusion_masked=np.stack(masked_fusion, axis=1),
            projector_unmasked=unmasked_projector,
            projector_masked=np.stack(masked_projector, axis=1),
            kappas=kappas,
        )


def _finetune_pass(
    model: LANISTRFinetuneModel,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, NDArray[np.float32] | NDArray[np.int64]]:
    model.eval()
    fusion: list[NDArray[np.float32]] = []
    logits: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = _move_batch(batch, device)
            fusion_cls = model.encode_fusion(device_batch)
            output_logits = model.classifier(fusion_cls)
            fusion.append(fusion_cls.cpu().numpy().astype(np.float32))
            logits.append(output_logits.cpu().numpy().astype(np.float32))
            labels.append(device_batch["labels"].cpu().numpy().astype(np.int64))
    return {
        "fusion": _concatenate(fusion),
        "logits": _concatenate(logits),
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
    }


def export_finetune_representations(
    model: LANISTRFinetuneModel,
    config: DictConfig,
    dataloaders: dict[str, DataLoader[dict[str, Tensor]]],
    output_dir: Path,
) -> None:
    device = model.device
    batch_size = int(config.training.get("eval_batch_size", config.training.batch_size))
    accuracies: dict[str, float] = {}
    for split in ("train", "test"):
        loader, original_transform = _loader_for_split(dataloaders, split, batch_size)
        result = _finetune_pass(model, loader, device)
        _restore_transform(dataloaders, split, original_transform)
        labels = cast(NDArray[np.int64], result["labels"])
        logits = cast(NDArray[np.float32], result["logits"])
        predictions = logits.argmax(axis=1)
        accuracies[split] = float(np.mean(predictions == labels))
        np.savez_compressed(
            output_dir / f"finetune_{split}.npz",
            indices=np.arange(labels.shape[0], dtype=np.int64),
            labels=labels,
            fusion=cast(NDArray[np.float32], result["fusion"]),
            logits=logits,
        )
    first_layer = cast(nn.Linear, model.classifier.classifier[0])
    output_layer = cast(nn.Linear, model.classifier.classifier[3])
    first_spectral = torch.linalg.matrix_norm(first_layer.weight.detach(), ord=2)
    output_rows = output_layer.weight.detach()
    pairwise_rows = output_rows[:, None, :] - output_rows[None, :, :]
    largest_pairwise_row = pairwise_rows.norm(dim=-1).max()
    pairwise_lipschitz_bound = float((first_spectral * largest_pairwise_row).cpu())
    metadata = {
        "dataset_type": str(config.dataset.dataset_type),
        "image": bool(config.dataset.image),
        "text": bool(config.dataset.text),
        "tab": bool(config.dataset.tab),
        "seed": int(config.seed),
        "ssl_objective": str(config.training.ssl_objective),
        "learnable_kappa": bool(config.riemannian.get("learnable_kappa", False)),
        "configured_kappa": float(config.riemannian.get("kappa", 0.0)),
        "train_accuracy": accuracies["train"],
        "test_accuracy": accuracies["test"],
        "classifier_first_spectral_norm": float(first_spectral.cpu()),
        "classifier_largest_pairwise_output_row_norm": float(largest_pairwise_row.cpu()),
        "classifier_pairwise_lipschitz_bound": pairwise_lipschitz_bound,
    }
    (output_dir / "mechanism_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

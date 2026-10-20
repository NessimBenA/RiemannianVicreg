import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import transformers


def _reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_signature(config: DictConfig) -> str:
    blob = json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


from .models.lanistr_model import (
    LANISTRPretrainModel,
    LANISTRFinetuneModel,
    LANISTRPretrainOutput,
    LANISTRFinetuneOutput,
    # RVICRegPretrainModel,  # DEPRECATED
    # RVICRegFinetuneModel,  # DEPRECATED
    # RVICRegPretrainOutput,  # DEPRECATED
)
from .mechanism_export import (
    export_finetune_representations,
    export_pretrain_representations,
)


logger = logging.getLogger(__name__)


class LANISTRTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        dataloaders: dict[str, DataLoader],
    ) -> None:
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.summary: dict[str, float] = {}

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        self.num_epochs = config.training.num_epochs
        self.lr = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.gradient_clip = config.training.gradient_clip
        self.warmup_epochs = config.training.warmup_epochs

        if config.training.get("output_dir"):
            self.output_dir = Path(config.training.output_dir)
        else:
            logger.warning(
                f"training.output_dir not set, using config.output_dir={config.output_dir}. "
                "This may cause checkpoint conflicts when running multiple experiments."
            )
            self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_metric = float("inf")

    def _get_optimizer_and_scheduler(
        self,
        num_training_steps: int,
        steps_per_epoch: int,
    ) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        kappa_lr_scale = self.config.riemannian.get("kappa_lr_scale", 0.1) if hasattr(self.config, "riemannian") else 0.1

        kappa_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "kappa" in name.lower():
                kappa_params.append(param)
            else:
                other_params.append(param)

        param_groups = [{"params": other_params, "lr": self.lr}]
        if kappa_params:
            param_groups.append({"params": kappa_params, "lr": self.lr * kappa_lr_scale})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )
        num_warmup_steps = self.warmup_epochs * steps_per_epoch
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        is_best: bool = False,
        prefix: str = "checkpoint",
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": self.config,
            "run_sig": _run_signature(self.config),
        }

        keep_all = self.config.training.get("keep_all_checkpoints", True)
        if keep_all:
            checkpoint_path = self.output_dir / f"{prefix}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.output_dir / f"{prefix}_best.pt"
            torch.save(checkpoint, best_path)

    def _load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        expected = _run_signature(self.config)
        got = checkpoint.get("run_sig")
        if got is not None and got != expected:
            raise RuntimeError(
                f"Checkpoint run_sig mismatch: expected {expected}, got {got}. "
                f"Likely overwritten/collision: {checkpoint_path}"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def train_supervised(self) -> None:
        logger.info("Starting supervised training")

        train_loader = self.dataloaders["train"]
        valid_loader = self.dataloaders.get("valid")

        steps_per_epoch = len(train_loader)
        num_training_steps = self.num_epochs * steps_per_epoch
        optimizer, scheduler = self._get_optimizer_and_scheduler(num_training_steps, steps_per_epoch)

        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            train_metrics = self._train_epoch_finetune(
                train_loader, optimizer, scheduler, epoch
            )

            val_metrics = None
            if valid_loader is not None:
                val_metrics = self._validate(valid_loader)

            is_best = False
            if val_metrics is not None and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                is_best = True

            self._save_checkpoint(epoch, optimizer, scheduler, is_best, "supervised")

            self._log_epoch_results(epoch, train_metrics, val_metrics, is_best)

    def train_simsiam(self) -> str:
        logger.info("Starting SimSiam pretraining")

        train_loader = self.dataloaders["train"]

        steps_per_epoch = len(train_loader)
        num_training_steps = self.num_epochs * steps_per_epoch
        optimizer, scheduler = self._get_optimizer_and_scheduler(num_training_steps, steps_per_epoch)

        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            train_metrics = self._train_epoch_pretrain(
                train_loader, optimizer, scheduler, epoch
            )

            is_best = train_metrics["loss"] < best_loss
            if is_best:
                best_loss = train_metrics["loss"]

            self._save_checkpoint(epoch, optimizer, scheduler, is_best, "pretrain")

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Loss: {train_metrics['loss']:.4f} "
                f"MLM: {train_metrics['mlm_loss']:.4f} "
                f"MIM: {train_metrics['mim_loss']:.4f} "
                f"MFM: {train_metrics['mfm_loss']:.4f} "
                f"MMM: {train_metrics['mmm_loss']:.4f}"
            )

        best_checkpoint_path = str(self.output_dir / "pretrain_best.pt")
        return best_checkpoint_path

    def train_simsiam_then_finetune(
        self,
        tabular_data_info: dict,
        num_classes: int,
    ) -> None:
        logger.info("Starting SimSiam pretraining followed by finetuning")

        pretrain_checkpoint = self.train_simsiam()

        logger.info("Loading pretrained model for finetuning")

        pretrain_model = self.model
        if not isinstance(pretrain_model, LANISTRPretrainModel):
            raise ValueError("Model must be LANISTRPretrainModel for simsiam_finetune mode")

        self._load_checkpoint(pretrain_checkpoint)

        if self.config.training.get("export_mechanism_representations", False):
            export_pretrain_representations(
                pretrain_model,
                self.config,
                self.dataloaders,
                self.output_dir,
            )

        _reset_seed(self.config.seed)
        logger.info(f"Reset seed to {self.config.seed} before finetuning for reproducibility")

        finetune_model = LANISTRFinetuneModel(
            config=self.config,
            tabular_data_info=tabular_data_info,
            num_classes=num_classes,
            pretrain_model=pretrain_model,
        )

        if self.config.training.get("freeze_encoders", True):
            finetune_model.freeze_encoders()

        self.model = finetune_model.to(self.device)

        self.train_supervised()

        if self.config.training.get("export_mechanism_representations", False):
            best_checkpoint = self.output_dir / "supervised_best.pt"
            if not best_checkpoint.exists():
                raise RuntimeError("No supervised best checkpoint was produced")
            self._load_checkpoint(str(best_checkpoint))
            export_finetune_representations(
                finetune_model,
                self.config,
                self.dataloaders,
                self.output_dir,
            )

    def _train_epoch_pretrain(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_mlm = 0.0
        total_mim = 0.0
        total_mfm = 0.0
        total_mmm = 0.0
        total_inv = 0.0
        total_var = 0.0
        total_cov = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}")

        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            output: LANISTRPretrainOutput = self.model(batch)
            loss = output.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mlm += output.loss_mlm.item() if output.loss_mlm is not None else 0
            total_mim += output.loss_mim.item() if output.loss_mim is not None else 0
            total_mfm += output.loss_mfm.item() if output.loss_mfm is not None else 0
            total_mmm += output.loss_mmm.item() if output.loss_mmm is not None else 0
            total_inv += output.loss_inv.item() if output.loss_inv is not None else 0
            total_var += output.loss_var.item() if output.loss_var is not None else 0
            total_cov += output.loss_cov.item() if output.loss_cov is not None else 0
            num_batches += 1

            pbar.set_postfix({"loss": loss.item()})

            pretrain_stats = {
                "kappa": output.kappa,
                "kappa_std": output.kappa_std,
                "kappa_min": output.kappa_min,
                "kappa_max": output.kappa_max,
                "boundary_proximity_max": output.boundary_proximity_max,
                "boundary_proximity_mean": output.boundary_proximity_mean,
                "tangent_norm_mean": output.tangent_norm_mean,
            }
            for name, value in pretrain_stats.items():
                if value is not None:
                    self.summary[f"pretrain/{name}"] = value.item()

            self.global_step += 1

        return {
            "loss": total_loss / num_batches,
            "mlm_loss": total_mlm / num_batches,
            "mim_loss": total_mim / num_batches,
            "mfm_loss": total_mfm / num_batches,
            "mmm_loss": total_mmm / num_batches,
            "inv_loss": total_inv / num_batches,
            "var_loss": total_var / num_batches,
            "cov_loss": total_cov / num_batches,
        }

    def _train_epoch_finetune(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch + 1}")

        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            output: LANISTRFinetuneOutput = self.model(batch)
            loss = output.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = output.logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": total_correct / total_samples,
            })

            self.global_step += 1

        num_batches = len(train_loader)
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output: LANISTRFinetuneOutput = self.model(batch)

            total_loss += output.loss.item()
            preds = output.logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)

        num_batches = len(val_loader)
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        test_loader = self.dataloaders.get("test")
        if test_loader is None:
            logger.warning("No test loader available")
            return {}

        best_checkpoint = self.output_dir / "supervised_best.pt"
        if best_checkpoint.exists():
            self._load_checkpoint(str(best_checkpoint))

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output: LANISTRFinetuneOutput = self.model(batch)

            total_loss += output.loss.item()
            preds = output.logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)

            probs = F.softmax(output.logits, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

        num_batches = len(test_loader)
        all_probs_np = np.concatenate(all_probs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)

        num_classes = all_probs_np.shape[1]
        if num_classes == 2:
            auroc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
        else:
            auroc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")

        metrics = {
            "loss": total_loss / num_batches,
            "accuracy": total_correct / total_samples,
            "auroc": auroc,
        }

        logger.info(f"Test Results: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}, AUROC={metrics['auroc']:.4f}")

        self.summary.update({f"test/{name}": float(value) for name, value in metrics.items()})

        return metrics

    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: Optional[dict[str, float]],
        is_best: bool,
    ) -> None:
        msg = f"Epoch {epoch + 1}/{self.num_epochs} - "
        msg += f"Train Loss: {train_metrics['loss']:.4f} "
        msg += f"Train Acc: {train_metrics['accuracy']:.4f}"

        if val_metrics is not None:
            msg += f" | Val Loss: {val_metrics['loss']:.4f} "
            msg += f"Val Acc: {val_metrics['accuracy']:.4f}"

        if is_best:
            msg += " [BEST]"

        logger.info(msg)

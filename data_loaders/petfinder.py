import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import omegaconf
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from .dataset_utils import (
    encode_tabular_features,
    get_image_transforms,
    get_text_tokenizer,
)

logger = logging.getLogger(__name__)


CATEGORICAL_COLS: List[str] = [
    "Type",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "Health",
    "State",
]
NUMERICAL_COLS: List[str] = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]


class PetfinderDataset(Dataset):
    def __init__(
        self,
        args: omegaconf.DictConfig,
        df: pd.DataFrame,
        transform: transforms.Compose,
        tokenizer: AutoTokenizer,
        feature_names: List[str],
        image_dir: Path,
        sentiment_dir: Path,
        text: bool = True,
        image: bool = True,
        tab: bool = True,
    ) -> None:
        self.args = args
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.image_dir = image_dir
        self.sentiment_dir = sentiment_dir

        self.text_enabled = text
        self.image_enabled = image
        self.tab_enabled = tab

        if tab:
            self.features = self.df[feature_names].values

    def __len__(self) -> int:
        return len(self.df)

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer.encode_plus(
            text,
            max_length=self.args.max_token_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def _load_sentiment(self, pet_id: str) -> str:
        sentiment_path = self.sentiment_dir / f"{pet_id}.json"
        if not sentiment_path.exists():
            return ""

        with open(sentiment_path, "r") as f:
            sentiment_data = json.load(f)

        sentences = sentiment_data.get("sentences", [])
        text_parts = [s["text"]["content"] for s in sentences]
        return " ".join(text_parts)

    def _load_image(self, pet_id: str) -> torch.Tensor:
        num_images = getattr(self.args, "num_images_per_pet", 1)
        available_images: List[Path] = []
        for i in range(1, num_images + 1):
            image_path = self.image_dir / f"{pet_id}-{i}.jpg"
            if image_path.exists():
                available_images.append(image_path)

        if available_images:
            selected_path = random.choice(available_images)
            img = Image.open(selected_path).convert("RGB")
            pixel_values = self.transform(img)
        else:
            pixel_values = torch.zeros(
                3, self.args.image_size, self.args.image_size, dtype=torch.float32
            )

        return pixel_values

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        pet_id = row["PetID"]
        item: Dict[str, torch.Tensor] = {}

        if self.text_enabled:
            description = (
                str(row["Description"]) if pd.notna(row["Description"]) else ""
            )
            sentiment_text = self._load_sentiment(pet_id)
            text = f"{description} {sentiment_text}".strip()
            encode_result = self.encode_text(text)
            item["input_ids"] = encode_result["input_ids"].squeeze(0)
            item["attention_mask"] = encode_result["attention_mask"].squeeze(0)

        if self.image_enabled:
            item["pixel_values"] = self._load_image(pet_id)

        if self.tab_enabled:
            item["features"] = torch.tensor(
                np.vstack(self.features[idx]).astype(np.float32)
            ).squeeze()

        item["labels"] = torch.tensor(row["AdoptionSpeed"], dtype=torch.long)

        return item


def get_train_val_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.05,
    seed: int = 2022,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(df))
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_test = int(len(df) * test_ratio)
    n_val = int(len(df) * val_ratio)
    n_train = len(df) - n_test - n_val

    train_df = df.iloc[indices[:n_train]].reset_index(drop=True)
    valid_df = df.iloc[indices[n_train : n_train + n_val]].reset_index(drop=True)
    test_df = df.iloc[indices[n_train + n_val :]].reset_index(drop=True)
    return train_df, valid_df, test_df


def load_petfinder(
    args: omegaconf.DictConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    root = Path(args.data_dir)

    possible_csv_paths = [
        root / "train" / "train.csv",
        root / "train.csv",
    ]

    csv_path = None
    for path in possible_csv_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find train.csv in any of: {[str(p) for p in possible_csv_paths]}"
        )

    if (root / "train_images").exists():
        image_dir = root / "train_images"
    else:
        image_dir = root / "images"

    if (root / "train_sentiment").exists():
        sentiment_dir = root / "train_sentiment"
    else:
        sentiment_dir = root / "sentiment"

    full_df = pd.read_csv(csv_path)

    if "AdoptionSpeed" not in full_df.columns:
        raise ValueError(
            "The PetFinder dataset must contain 'AdoptionSpeed' column for training. "
            "The test set from Kaggle does not include labels."
        )

    test_ratio = getattr(args, "test_ratio", 0.15)
    val_ratio = getattr(args, "val_ratio", 0.05)

    train_df, valid_df, test_df = get_train_val_test_split(
        full_df, test_ratio=test_ratio, val_ratio=val_ratio
    )

    feature_names = CATEGORICAL_COLS + NUMERICAL_COLS

    train_df, cat_idxs, cat_dims, input_dim, encoders, num_stats = encode_tabular_features(
        train_df.copy(), CATEGORICAL_COLS, NUMERICAL_COLS
    )
    valid_df, _, _, _, _, _ = encode_tabular_features(
        valid_df.copy(), CATEGORICAL_COLS, NUMERICAL_COLS, encoders, num_stats
    )
    test_df, _, _, _, _, _ = encode_tabular_features(
        test_df.copy(), CATEGORICAL_COLS, NUMERICAL_COLS, encoders, num_stats
    )

    train_transform, test_transform = get_image_transforms(args)

    text_enabled = getattr(args, "text", True)
    image_enabled = getattr(args, "image", True)
    tab_enabled = getattr(args, "tab", True)

    mm_train = PetfinderDataset(
        args=args,
        df=train_df,
        transform=train_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        image_dir=image_dir,
        sentiment_dir=sentiment_dir,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    mm_valid = PetfinderDataset(
        args=args,
        df=valid_df,
        transform=test_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        image_dir=image_dir,
        sentiment_dir=sentiment_dir,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    mm_test = PetfinderDataset(
        args=args,
        df=test_df,
        transform=test_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        image_dir=image_dir,
        sentiment_dir=sentiment_dir,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    num_classes = len(full_df["AdoptionSpeed"].unique())

    tabular_data_information = {
        "input_dim": input_dim,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "feature_names": feature_names,
        "num_classes": num_classes,
    }

    logger.info("PetFinder dataset loaded:")
    logger.info("  Train samples: %d", len(mm_train))
    logger.info("  Valid samples: %d", len(mm_valid))
    logger.info("  Test samples: %d", len(mm_test))

    return {
        "train": mm_train,
        "valid": mm_valid,
        "test": mm_test,
        "tabular_data_information": tabular_data_information,
    }

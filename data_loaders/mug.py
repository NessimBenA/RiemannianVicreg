from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "csgo_skin_quality": {
        "folder": "CSGO-Skin-quality",
        "label_col": "Skin Quality",
        "text_col": "Skin Name",
        "image_col": "Image Path",
        "categorical_cols": ["Availability", "Skin Category"],
        "numerical_cols": ["Min Price", "Max Price"],
    },
    "hearthstone_all_cardclass": {
        "folder": "Hearthstone-All-cardClass",
        "label_col": "cardClass",
        "text_col": "text",
        "image_col": "Image Path",
        "categorical_cols": ["rarity", "type"],
        "numerical_cols": ["attack", "cost", "health"],
    },
    "hearthstone_all_set": {
        "folder": "Hearthstone-All-set",
        "label_col": "set",
        "text_col": "text",
        "image_col": "Image Path",
        "categorical_cols": ["rarity", "type", "cardClass"],
        "numerical_cols": ["attack", "cost", "health"],
    },
    "hearthstone_minion_race": {
        "folder": "Hearthstone-Minion-race",
        "label_col": "race",
        "text_col": "text",
        "image_col": "Image Path",
        "categorical_cols": ["rarity", "set", "cardClass"],
        "numerical_cols": ["attack", "cost", "health"],
    },
    "hearthstone_spell_spellschool": {
        "folder": "Hearthstone-Spell-spellSchool",
        "label_col": "spellSchool",
        "text_col": "text",
        "image_col": "Image Path",
        "categorical_cols": ["rarity", "set", "cardClass"],
        "numerical_cols": ["cost"],
    },
    "leagueoflegends_skin_category": {
        "folder": "LeagueOfLegends-Skin-category",
        "label_col": "Category",
        "text_col": "SkinName",
        "image_col": "Image Path",
        "categorical_cols": ["Concept", "Sold ingame?"],
        "numerical_cols": ["Price"],
    },
    "pokemon_primary_type": {
        "folder": "Pokemon-primary_type",
        "label_col": "type_1",
        "text_col": "name",
        "image_col": "Image Path",
        "categorical_cols": ["generation", "status"],
        "numerical_cols": [
            "hp",
            "attack",
            "defense",
            "sp_attack",
            "sp_defense",
            "speed",
        ],
    },
    "pokemon_secondary_type": {
        "folder": "Pokemon-secondary_type",
        "label_col": "type_2",
        "text_col": "name",
        "image_col": "Image Path",
        "categorical_cols": ["generation", "status"],
        "numerical_cols": [
            "hp",
            "attack",
            "defense",
            "sp_attack",
            "sp_defense",
            "speed",
        ],
    },
}


class MuGDataset(Dataset):
    def __init__(
        self,
        args: omegaconf.DictConfig,
        df: pd.DataFrame,
        transform: transforms.Compose,
        tokenizer: AutoTokenizer,
        feature_names: List[str],
        text_col: str,
        image_col: str,
        label_col: str,
        dataset_folder: Path,
        label_encoder: LabelEncoder,
        text: bool = True,
        image: bool = True,
        tab: bool = True,
    ) -> None:
        self.args = args
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.dataset_folder = dataset_folder
        self.label_encoder = label_encoder

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        item: Dict[str, torch.Tensor] = {}

        if self.text_enabled:
            text = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
            encode_result = self.encode_text(text)
            item["input_ids"] = encode_result["input_ids"].squeeze(0)
            item["attention_mask"] = encode_result["attention_mask"].squeeze(0)

        if self.image_enabled:
            image_col = (
                self.image_col if self.image_col in self.df.columns else "image_path"
            )
            image_path = self.dataset_folder / row[image_col]
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                pixel_values = self.transform(img)
            else:
                pixel_values = torch.zeros(
                    3, self.args.image_size, self.args.image_size, dtype=torch.float32
                )
            item["pixel_values"] = pixel_values

        if self.tab_enabled:
            item["features"] = torch.tensor(
                np.vstack(self.features[idx]).astype(np.float32)
            ).squeeze()

        item["labels"] = torch.tensor(row["encoded_label"], dtype=torch.long)

        return item


def load_mug(
    args: omegaconf.DictConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    dataset_type = args.dataset_type
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    config = DATASET_CONFIGS[dataset_type]
    dataset_folder = Path(args.data_dir) / config["folder"]

    categorical_cols = config["categorical_cols"]
    numerical_cols = config["numerical_cols"]
    feature_names = categorical_cols + numerical_cols

    split_map = {"train": "train", "valid": "dev", "test": "test"}

    train_df = pd.read_csv(dataset_folder / f"{split_map['train']}.csv")
    valid_df = pd.read_csv(dataset_folder / f"{split_map['valid']}.csv")
    test_df = pd.read_csv(dataset_folder / f"{split_map['test']}.csv")

    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(full_df[config["label_col"]])

    train_df["encoded_label"] = label_encoder.transform(train_df[config["label_col"]])
    valid_df["encoded_label"] = label_encoder.transform(valid_df[config["label_col"]])
    test_df["encoded_label"] = label_encoder.transform(test_df[config["label_col"]])

    train_df, cat_idxs, cat_dims, input_dim, encoders, num_stats = encode_tabular_features(
        train_df.copy(), categorical_cols, numerical_cols
    )
    valid_df, _, _, _, _, _ = encode_tabular_features(
        valid_df.copy(), categorical_cols, numerical_cols, encoders, num_stats
    )
    test_df, _, _, _, _, _ = encode_tabular_features(
        test_df.copy(), categorical_cols, numerical_cols, encoders, num_stats
    )

    train_transform, test_transform = get_image_transforms(args)

    text_enabled = getattr(args, "text", True)
    image_enabled = getattr(args, "image", True)
    tab_enabled = getattr(args, "tab", True)

    mm_train = MuGDataset(
        args=args,
        df=train_df,
        transform=train_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        text_col=config["text_col"],
        image_col=config["image_col"],
        label_col=config["label_col"],
        dataset_folder=dataset_folder,
        label_encoder=label_encoder,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    mm_valid = MuGDataset(
        args=args,
        df=valid_df,
        transform=test_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        text_col=config["text_col"],
        image_col=config["image_col"],
        label_col=config["label_col"],
        dataset_folder=dataset_folder,
        label_encoder=label_encoder,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    mm_test = MuGDataset(
        args=args,
        df=test_df,
        transform=test_transform,
        tokenizer=tokenizer,
        feature_names=feature_names,
        text_col=config["text_col"],
        image_col=config["image_col"],
        label_col=config["label_col"],
        dataset_folder=dataset_folder,
        label_encoder=label_encoder,
        text=text_enabled,
        image=image_enabled,
        tab=tab_enabled,
    )

    tabular_data_information = {
        "input_dim": input_dim,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "feature_names": feature_names,
        "num_classes": len(label_encoder.classes_),
    }

    return {
        "train": mm_train,
        "valid": mm_valid,
        "test": mm_test,
        "tabular_data_information": tabular_data_information,
    }

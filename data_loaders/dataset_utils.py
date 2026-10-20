from typing import Dict, List, Optional, Tuple

import numpy as np
import omegaconf
import pandas as pd
import torch
from sklearn import preprocessing
from torchvision import transforms
from transformers import AutoTokenizer, ViTImageProcessor


class ImageMaskGenerator:
    def __init__(
        self,
        input_size: int = 224,
        mask_patch_size: int = 16,
        model_patch_size: int = 16,
        mask_ratio: float = 0.5,
    ) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self) -> torch.Tensor:
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten(), dtype=torch.bool)


def get_image_transforms(
    args: omegaconf.DictConfig,
) -> Tuple[transforms.Compose, transforms.Compose]:
    image_processor = ViTImageProcessor.from_pretrained(args.image_encoder_name)

    image_size = args.image_size
    image_crop = getattr(args, "image_crop", args.image_size)

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_crop),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            ),
        ]
    )

    return train_transforms, test_transforms


def get_text_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def clean_numeric_column(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace("$", "", regex=False)
        cleaned = cleaned.str.replace(",", "", regex=False)
        cleaned = cleaned.str.strip()
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        return cleaned
    return series


def encode_tabular_features(
    data: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    encoders: Optional[Dict[str, preprocessing.LabelEncoder]] = None,
    numerical_stats: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[int], List[int], int, Dict[str, preprocessing.LabelEncoder], Dict[str, float]]:
    categorical_columns: List[str] = []
    categorical_dims: Dict[str, int] = {}
    fitted_encoders: Dict[str, preprocessing.LabelEncoder] = encoders if encoders is not None else {}
    fitted_numerical_stats: Dict[str, float] = numerical_stats if numerical_stats is not None else {}

    for col in numerical_cols:
        if col in data.columns:
            data.loc[:, col] = clean_numeric_column(data[col])
            if col in fitted_numerical_stats:
                col_mean = fitted_numerical_stats[col]
            else:
                col_mean = data[col].mean()
                fitted_numerical_stats[col] = col_mean
            data.loc[:, col] = data.loc[:, col].fillna(col_mean)

    for col in data.columns:
        if col in categorical_cols:
            data[col] = data[col].astype(str).replace("nan", "VV_likely")
            if col in fitted_encoders:
                l_enc = fitted_encoders[col]
                known_classes = set(l_enc.classes_)
                data[col] = data[col].apply(
                    lambda x, kc=known_classes: x if x in kc else "VV_likely"
                )
                data[col] = l_enc.transform(data[col].values)
            else:
                l_enc = preprocessing.LabelEncoder()
                unique_values = list(data[col].unique())
                if "VV_likely" not in unique_values:
                    unique_values.append("VV_likely")
                l_enc.fit(unique_values)
                data[col] = l_enc.transform(data[col].values)
                fitted_encoders[col] = l_enc
            categorical_columns.append(col)
            categorical_dims[col] = len(fitted_encoders[col].classes_)

    feature_names = categorical_cols + numerical_cols
    cat_idxs = [i for i, col in enumerate(feature_names) if col in categorical_columns]
    cat_dims = [categorical_dims[feature_names[i]] for i in cat_idxs]
    input_dim = len(feature_names)

    return data, cat_idxs, cat_dims, input_dim, fitted_encoders, fitted_numerical_stats

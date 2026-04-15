from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


SEED = 3407
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = [
    "android_25c.SG",
    "christianity_25c.SG",
    "douban_25c.SG",
    "twitter_25c.SG",
]
DATASET_NAMES = ["android", "christianity", "douban", "twitter"]
DATASET_IDX = 1

RECALL_K_VALUES = (5, 15, 25)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.25
    residual: bool = True
    root_hidden_dim: int = 64
    pair_hidden_dim: int = 64
    count_hidden_dim: int = 64


@dataclass(frozen=True)
class LossConfig:
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    ranking_margin: float = 0.3
    ranking_loss_weight: float = 1.0
    compat1_loss_weight: float = 0.5
    compat2_loss_weight: float = 0.5
    count_loss_weight: float = 0.2
    hard_negative_topk: int = 8
    decoder_step_loss_weight: float = 0.6
    decoder_set_loss_weight: float = 0.35


@dataclass(frozen=True)
class DecoderConfig:
    candidate_pool_cap: int = 30
    candidate_pool_multiplier: int = 4
    candidate_pool_bias: int = 12
    unary_weight: float = 1.0
    coverage_weight: float = 0.8
    compatibility_weight: float = 0.35
    count_prior_weight: float = 0.25
    decoder_hidden_dim: int = 64
    step_temperature: float = 0.7
    set_margin: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    warmup_epochs: int = 100  # warmup 期间仅优化 unary/pair/count 辅助任务
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    count_threshold: float = 0.5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    allow_zero_source_in_maxcc: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset_idx: int = DATASET_IDX
    eval_scope: str = "maxcc"
    include_ring_hops: int = 1
    max_boundary_ring_size: int | None = 512
    skip_twitter: bool = True
    max_count_cap: int | None = None
    quick_train_samples: int | None = None
    quick_val_samples: int | None = None
    quick_test_samples: int | None = None


@dataclass(frozen=True)
class SelectionConfig:
    recall_k: int = 5
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "map": 0.35,
            "p@k_true": 0.30,
            "recall@k": 0.20,
            "f1": 0.15,
            "aed_gain": 0.10,
            "count_gain": 0.10,
        }
    )


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    @property
    def dataset_name(self) -> str:
        return DATASET_NAMES[self.data.dataset_idx]

    @property
    def dataset_file(self) -> str:
        return DATASETS[self.data.dataset_idx]


def build_experiment_config() -> ExperimentConfig:
    config = ExperimentConfig()
    if config.data.skip_twitter and DATASET_NAMES[config.data.dataset_idx] == "twitter":
        raise ValueError("The current implementation intentionally skips the twitter dataset.")
    return config

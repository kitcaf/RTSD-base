from __future__ import annotations

import logging
import os
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_training_logger(log_dir: str = "history", log_name: str = "training_log.txt") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(f"rtsd_training_{log_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_print(logger: logging.Logger | None, message: str) -> None:
    print(message)
    if logger is not None:
        logger.info(message)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def split_samples_by_cascade(
    samples: Sequence,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    cascade_ids = sorted({int(sample.cascade_id) for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(cascade_ids)

    train_end = int(len(cascade_ids) * train_ratio)
    val_end = int(len(cascade_ids) * (train_ratio + val_ratio))

    train_ids = set(cascade_ids[:train_end])
    val_ids = set(cascade_ids[train_end:val_end])
    test_ids = set(cascade_ids[val_end:])

    train_samples = [sample for sample in samples if int(sample.cascade_id) in train_ids]
    val_samples = [sample for sample in samples if int(sample.cascade_id) in val_ids]
    test_samples = [sample for sample in samples if int(sample.cascade_id) in test_ids]
    return train_samples, val_samples, test_samples


def slice_if_needed(samples: Sequence, limit: int | None) -> List:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return list(samples)
    return list(samples[:limit])


def compute_selection_score(metrics: dict, recall_k: int, weights: dict) -> float:
    recall_key = f"recall@{recall_k}"
    aed_gain = 1.0 / (1.0 + max(float(metrics.get("aed", 0.0)), 0.0))
    count_gain = 1.0 / (1.0 + max(float(metrics.get("count_mae", 0.0)), 0.0))
    return (
        weights.get("map", 0.0) * float(metrics.get("map", 0.0)) +
        weights.get("p@k_true", 0.0) * float(metrics.get("p@k_true", 0.0)) +
        weights.get("recall@k", 0.0) * float(metrics.get(recall_key, 0.0)) +
        weights.get("f1", 0.0) * float(metrics.get("f1", 0.0)) +
        weights.get("aed_gain", 0.0) * aed_gain +
        weights.get("count_gain", 0.0) * count_gain
    )


def count_metrics(pred_counts: Iterable[int], true_counts: Iterable[int]) -> dict:
    predicted = np.asarray(list(pred_counts), dtype=np.int64)
    truth = np.asarray(list(true_counts), dtype=np.int64)
    if predicted.size == 0:
        return {"count_mae": 0.0, "count_acc": 0.0, "count_within_1": 0.0}
    errors = np.abs(predicted - truth)
    return {
        "count_mae": float(errors.mean()),
        "count_acc": float((errors == 0).mean()),
        "count_within_1": float((errors <= 1).mean()),
    }

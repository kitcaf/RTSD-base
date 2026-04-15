from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from config import ExperimentConfig
from metrics_utils import calculate_aed, calculate_map, calculate_precision_at_k
from utils import compute_selection_score, count_metrics, log_print, safe_mean
from .data_types import ContextGraphSample
from .decoder import SetDecoder
from .feature_builder import FeatureBuilder
from .network import ContextRootModel, ModelOutput
from .score_adapter import ScoreAdapter


@dataclass(frozen=True)
class TrainerArtifacts:
    model: ContextRootModel
    best_epoch: int
    best_val_score: float


class Trainer:
    def __init__(self, config: ExperimentConfig, device: torch.device, dist_matrix, logger=None) -> None:
        self.config = config
        self.device = device
        self.dist_matrix = dist_matrix
        self.decoder: SetDecoder | None = None
        self.score_adapter = ScoreAdapter()
        self.logger = logger

    def build_model(self, train_samples: Sequence[ContextGraphSample]) -> ContextRootModel:
        max_count = max(sample.k_label for sample in train_samples)
        if self.config.data.max_count_cap is not None:
            max_count = min(max_count, self.config.data.max_count_cap)
        max_count = max(max_count, 1)
        pair_feature_dim = (
            int(train_samples[0].pair1_features.size(-1))
            if train_samples[0].pair1_features.numel() > 0
            else len(FeatureBuilder.PAIR_FEATURE_NAMES)
        )
        model = ContextRootModel(
            num_node_features=int(train_samples[0].x.size(-1)),
            pair_feature_dim=pair_feature_dim,
            max_count=max_count,
            config=self.config.model,
        ).to(self.device)
        self.decoder = SetDecoder(
            config=self.config.decoder,
            hidden_dim=self.config.model.hidden_dim,
            node_feature_dim=int(train_samples[0].x.size(-1)),
        ).to(self.device)
        return model

    def train(
        self,
        model: ContextRootModel,
        train_samples: Sequence[ContextGraphSample],
        val_samples: Sequence[ContextGraphSample],
    ) -> TrainerArtifacts:
        if self.decoder is None:
            raise RuntimeError("Decoder must be initialized via build_model before training.")
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.decoder.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        loader = DataLoader(list(train_samples), batch_size=self.config.training.batch_size, shuffle=True, collate_fn=list)

        best_model_state_dict = None
        best_decoder_state_dict = None
        best_epoch = 0
        best_val_score = float("-inf")

        for epoch in range(self.config.training.epochs):
            model.train()
            self.decoder.train()
            warmup_only = epoch < self.config.training.warmup_epochs
            epoch_loss_values = []
            for batch_samples in loader:
                optimizer.zero_grad()
                batch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                valid_count = 0
                for sample in batch_samples:
                    sample = sample.to(self.device)
                    outputs = model(sample)
                    batch_loss = batch_loss + self.compute_total_loss(sample, outputs, warmup_only=warmup_only)
                    valid_count += 1
                if valid_count == 0:
                    continue
                batch_loss = batch_loss / valid_count
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(self.decoder.parameters()),
                    self.config.training.gradient_clip_norm,
                )
                optimizer.step()
                epoch_loss_values.append(float(batch_loss.item()))

            val_metrics = self.evaluate(model, val_samples)
            val_score = compute_selection_score(
                metrics=val_metrics,
                recall_k=self.config.selection.recall_k,
                weights=self.config.selection.metric_weights,
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch + 1
                best_model_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_decoder_state_dict = {
                    key: value.detach().cpu().clone() for key, value in self.decoder.state_dict().items()
                }

            log_print(
                self.logger,
                (
                    f"Epoch {epoch + 1:03d} | loss={safe_mean(epoch_loss_values):.4f} | "
                    f"val_score={val_score:.4f} | f1={val_metrics['f1']:.4f} | "
                    f"map={val_metrics['map']:.4f} | p@k={val_metrics['p@k_true']:.4f} | "
                    f"aed={val_metrics['aed']:.4f} | count_mae={val_metrics['count_mae']:.4f}"
                ),
            )

        if best_model_state_dict is None or best_decoder_state_dict is None:
            raise RuntimeError("training finished without a usable checkpoint")
        model.load_state_dict(best_model_state_dict)
        self.decoder.load_state_dict(best_decoder_state_dict)
        return TrainerArtifacts(
            model=model,
            best_epoch=best_epoch,
            best_val_score=best_val_score,
        )

    def compute_total_loss(self, sample: ContextGraphSample, outputs: ModelOutput, warmup_only: bool) -> torch.Tensor:
        task_losses = self.compute_task_losses(sample, outputs)
        total_loss = task_losses["root"] + self.config.losses.count_loss_weight * task_losses["count"]
        total_loss = total_loss + self.config.losses.ranking_loss_weight * task_losses["rank"]
        if not warmup_only:
            total_loss = total_loss + self.config.losses.compat1_loss_weight * task_losses["compat1"]
            total_loss = total_loss + self.config.losses.compat2_loss_weight * task_losses["compat2"]
            total_loss = total_loss + self.config.losses.decoder_step_loss_weight * task_losses["decoder_step"]
            total_loss = total_loss + self.config.losses.decoder_set_loss_weight * task_losses["decoder_set"]
        return total_loss

    def compute_task_losses(self, sample: ContextGraphSample, outputs: ModelOutput) -> dict:
        if self.decoder is None:
            raise RuntimeError("Decoder must be initialized before computing losses.")
        decoder_losses = self.decoder.compute_losses(
            sample=sample,
            hidden_states=outputs.hidden_states,
            node_logits=outputs.node_logits,
            compat1_logits=outputs.compat1_logits,
            compat2_logits=outputs.compat2_logits,
            count_logits=outputs.count_logits,
        )
        return {
            "root": self._root_loss(sample, outputs.node_logits),
            "rank": self._rank_loss(sample, outputs.node_logits),
            "compat1": self._compat1_loss(sample, outputs.compat1_logits),
            "compat2": self._compat2_loss(sample, outputs.compat2_logits),
            "count": self._count_loss(sample, outputs.count_logits),
            "decoder_step": decoder_losses.step_loss,
            "decoder_set": decoder_losses.set_loss,
        }

    def _root_loss(self, sample: ContextGraphSample, node_logits: torch.Tensor) -> torch.Tensor:
        mask = sample.loss_mask
        logits = node_logits[mask]
        targets = sample.y_root[mask]
        if logits.numel() == 0:
            return node_logits.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha = torch.where(targets > 0.5, self.config.losses.focal_alpha, 1.0 - self.config.losses.focal_alpha)
        focal_weight = alpha * (1.0 - pt).pow(self.config.losses.focal_gamma)
        return (focal_weight * bce).mean()

    def _rank_loss(self, sample: ContextGraphSample, node_logits: torch.Tensor) -> torch.Tensor:
        maxcc_positions = sample.maxcc_local_idx
        labels = sample.y_root[maxcc_positions]
        positives = maxcc_positions[labels > 0.5]
        negatives = maxcc_positions[labels <= 0.5]
        if positives.numel() == 0 or negatives.numel() == 0:
            return node_logits.new_zeros(())

        heuristic_indices = FeatureBuilder.NODE_FEATURE_INDEX
        maxcc_features = sample.x[maxcc_positions]
        negative_mask = labels <= 0.5
        hard_negative_scores = torch.stack(
            [
                node_logits[negatives].detach(),
                maxcc_features[negative_mask, heuristic_indices["infected_degree_in_maxcc"]],
                maxcc_features[negative_mask, heuristic_indices["closeness_in_maxcc"]],
            ],
            dim=0,
        ).mean(dim=0)
        topk = min(self.config.losses.hard_negative_topk, negatives.numel())
        _, top_indices = torch.topk(hard_negative_scores, k=topk)
        hard_negatives = negatives[top_indices]

        positive_logits = node_logits[positives]
        negative_logits = node_logits[hard_negatives]
        pairwise_margin = self.config.losses.ranking_margin - (positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0))
        return torch.relu(pairwise_margin).mean()

    def _compat1_loss(self, sample: ContextGraphSample, compat1_logits: torch.Tensor) -> torch.Tensor:
        if compat1_logits.numel() == 0 or sample.pair1_compat_mask.sum().item() == 0:
            return sample.x.new_zeros(())
        return F.binary_cross_entropy_with_logits(
            compat1_logits[sample.pair1_compat_mask],
            sample.pair1_compat_label[sample.pair1_compat_mask],
        )

    def _compat2_loss(self, sample: ContextGraphSample, compat2_logits: torch.Tensor) -> torch.Tensor:
        if compat2_logits.numel() == 0 or sample.pair2_compat_mask.sum().item() == 0:
            return sample.x.new_zeros(())
        return F.binary_cross_entropy_with_logits(
            compat2_logits[sample.pair2_compat_mask],
            sample.pair2_compat_label[sample.pair2_compat_mask],
        )

    def _count_loss(self, sample: ContextGraphSample, count_logits: torch.Tensor) -> torch.Tensor:
        target = torch.zeros_like(count_logits)
        count_cap = min(sample.k_label, count_logits.numel())
        if count_cap > 0:
            target[:count_cap] = 1.0
        return F.binary_cross_entropy_with_logits(count_logits, target)

    def evaluate(
        self,
        model: ContextRootModel,
        samples: Sequence[ContextGraphSample],
    ) -> dict:
        if self.decoder is None:
            raise RuntimeError("Decoder must be initialized before evaluation.")
        model.eval()
        self.decoder.eval()
        precision_values: List[float] = []
        recall_values: List[float] = []
        f1_values: List[float] = []
        auc_values: List[float] = []
        recall_at_k = {k: [] for k in (5, 15, 25)}
        map_values: List[float] = []
        pk_values: List[float] = []
        aed_values: List[float] = []
        predicted_counts: List[int] = []
        true_counts: List[int] = []

        with torch.no_grad():
            for sample in samples:
                sample_on_device = sample.to(self.device)
                outputs = model(sample_on_device)
                decode_result = self.decoder.decode(
                    sample=sample_on_device,
                    hidden_states=outputs.hidden_states,
                    node_logits=outputs.node_logits,
                    compat1_logits=outputs.compat1_logits,
                    compat2_logits=outputs.compat2_logits,
                    count_logits=outputs.count_logits,
                )
                adapted = self.score_adapter.adapt(
                    sample=sample,
                    selected_nodes=decode_result.selected_nodes,
                    boosted_scores=decode_result.node_scores,
                )
                predicted_counts.append(decode_result.selected_count)
                true_counts.append(adapted.true_count)

                precision_values.append(precision_score(adapted.y_true, adapted.y_pred, zero_division=0))
                recall_values.append(recall_score(adapted.y_true, adapted.y_pred, zero_division=0))
                f1_values.append(f1_score(adapted.y_true, adapted.y_pred, zero_division=0))

                try:
                    if len(np.unique(adapted.y_true)) > 1:
                        auc_values.append(roc_auc_score(adapted.y_true, adapted.y_score))
                except ValueError:
                    pass

                if adapted.true_count > 0:
                    ranking_size = len(adapted.y_score)
                    sorted_indices = np.argsort(-adapted.y_score)
                    for top_k in (5, 15, 25):
                        top_positions = sorted_indices[:top_k]
                        recall_at_k[top_k].append(float(adapted.y_true[top_positions].sum() / adapted.true_count))

                    map_values.append(calculate_map(adapted.y_score, adapted.y_true, ranking_size))
                    pk_values.append(calculate_precision_at_k(adapted.y_score, adapted.y_true, adapted.true_count))
                    aed_values.append(
                        calculate_aed(
                            adapted.y_score,
                            adapted.y_true,
                            dist_matrix=self.dist_matrix,
                            top_k=adapted.true_count,
                            node_indices=adapted.node_indices,
                        )
                    )

        metrics = {
            "auc": safe_mean(auc_values),
            "precision": safe_mean(precision_values),
            "recall": safe_mean(recall_values),
            "f1": safe_mean(f1_values),
            "map": safe_mean(map_values),
            "p@k_true": safe_mean(pk_values),
            "aed": safe_mean(aed_values),
        }
        for top_k, values in recall_at_k.items():
            metrics[f"recall@{top_k}"] = safe_mean(values)
        metrics.update(count_metrics(predicted_counts, true_counts))
        return metrics

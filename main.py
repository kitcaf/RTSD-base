from __future__ import annotations

from config import DEVICE, RECALL_K_VALUES, SEED, build_experiment_config
from data_loader import load_raw_data
from metrics_utils import precompute_shortest_paths
from model import ContextGraphDatasetBuilder, Trainer
from utils import log_print, setup_seed, setup_training_logger, slice_if_needed, split_samples_by_cascade


def main() -> None:
    config = build_experiment_config()
    setup_seed(SEED)

    logger = setup_training_logger(log_name=f"{config.dataset_name}_context_gatv2.log")
    log_print(logger, "=" * 60)
    log_print(logger, "[*] 实验: ContextGraph + GATv2 + 可学习 Set Decoder")
    log_print(logger, f"[*] 数据集: {config.dataset_name}")
    log_print(logger, f"[*] 设备: {DEVICE}")
    log_print(logger, f"[*] Recall@K: {RECALL_K_VALUES}")
    log_print(logger, f"[*] Eval Scope: {config.data.eval_scope}")
    log_print(logger, f"[*] Warmup Epochs: {config.training.warmup_epochs}")
    log_print(logger, "=" * 60)

    adjacency_matrix, influence_matrices = load_raw_data(config.dataset_file)
    dataset_builder = ContextGraphDatasetBuilder(
        adjacency_matrix=adjacency_matrix,
        include_ring_hops=config.data.include_ring_hops,
        max_boundary_ring_size=config.data.max_boundary_ring_size,
        allow_zero_source_in_maxcc=config.training.allow_zero_source_in_maxcc,
        seed=SEED,
    )
    samples, stats = dataset_builder.build_dataset(influence_matrices)
    if not samples:
        raise RuntimeError("No valid samples were built. Please check dataset construction settings.")

    log_print(
        logger,
        f"[*] 样本构建完成: kept={stats.kept_samples}, skipped_empty={stats.skipped_empty}, "
        f"skipped_no_source_in_maxcc={stats.skipped_without_source_in_maxcc}",
    )

    train_samples, val_samples, test_samples = split_samples_by_cascade(
        samples=samples,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        seed=SEED,
    )
    train_samples = slice_if_needed(train_samples, config.data.quick_train_samples)
    val_samples = slice_if_needed(val_samples, config.data.quick_val_samples)
    test_samples = slice_if_needed(test_samples, config.data.quick_test_samples)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Train/val/test split produced an empty subset.")

    log_print(
        logger,
        f"[*] 数据划分: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}",
    )

    log_print(logger, "[*] 预计算 AED 懒加载最短路查询器...")
    dist_matrix = precompute_shortest_paths(adjacency_matrix)

    trainer = Trainer(config=config, device=DEVICE, dist_matrix=dist_matrix, logger=logger)
    model = trainer.build_model(train_samples)
    training_artifacts = trainer.train(model=model, train_samples=train_samples, val_samples=val_samples)
    test_metrics = trainer.evaluate(
        model=training_artifacts.model,
        samples=test_samples,
    )

    log_print(logger, "\n" + "=" * 60)
    log_print(logger, f"[*] 最佳 Epoch: {training_artifacts.best_epoch}")
    log_print(logger, f"[*] 最佳验证分数: {training_artifacts.best_val_score:.4f}")
    log_print(logger, f"[*] 测试 AUC       : {test_metrics['auc']:.4f}")
    log_print(logger, f"[*] 测试 Precision : {test_metrics['precision']:.4f}")
    log_print(logger, f"[*] 测试 Recall    : {test_metrics['recall']:.4f}")
    log_print(logger, f"[*] 测试 F1        : {test_metrics['f1']:.4f}")
    for recall_k in RECALL_K_VALUES:
        log_print(logger, f"[*] 测试 Recall@{recall_k:<2} : {test_metrics[f'recall@{recall_k}']:.4f}")
    log_print(logger, f"[*] 测试 MAP       : {test_metrics['map']:.4f}")
    log_print(logger, f"[*] 测试 P@K_true  : {test_metrics['p@k_true']:.4f}")
    log_print(logger, f"[*] 测试 AED       : {test_metrics['aed']:.4f}")
    log_print(logger, f"[*] Count MAE      : {test_metrics['count_mae']:.4f}")
    log_print(logger, f"[*] Count Acc      : {test_metrics['count_acc']:.4f}")
    log_print(logger, f"[*] Count Within 1 : {test_metrics['count_within_1']:.4f}")
    log_print(logger, "=" * 60)


if __name__ == "__main__":
    main()

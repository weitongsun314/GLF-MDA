import random
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_edge_list_matrix(path: str, one_based: bool = True) -> np.ndarray:
    """
    读取 adj.txt 三列边列表：
    disease_id, microbe_id, value
    输出 disease × microbe 的 0/1 矩阵
    """
    edge_list = np.loadtxt(path, dtype=int)
    if edge_list.ndim == 1:
        edge_list = edge_list.reshape(1, -1)

    disease_idx = edge_list[:, 0].astype(int)
    microbe_idx = edge_list[:, 1].astype(int)
    values = edge_list[:, 2].astype(np.float32)

    if one_based:
        disease_idx = disease_idx - 1
        microbe_idx = microbe_idx - 1

    n_disease = disease_idx.max() + 1
    n_microbe = microbe_idx.max() + 1

    assoc_matrix = np.zeros((n_disease, n_microbe), dtype=np.float32)
    assoc_matrix[disease_idx, microbe_idx] = values

    return assoc_matrix


def compute_gip_similarity(interaction_matrix: np.ndarray) -> np.ndarray:
    """
    对“行向量”计算 GIP kernel similarity。
    输入若是 disease × microbe，则输出 disease GIP；
    输入若是 microbe × disease，则输出 microbe GIP。
    """
    interaction_matrix = interaction_matrix.astype(np.float32)

    squared_norm = np.sum(interaction_matrix ** 2, axis=1)
    nonzero_norm = squared_norm[squared_norm > 0]

    if nonzero_norm.size == 0:
        gamma = 1.0
    else:
        gamma = 1.0 / np.mean(nonzero_norm)

    gram = interaction_matrix @ interaction_matrix.T
    sq_dist = squared_norm[:, None] + squared_norm[None, :] - 2.0 * gram
    sq_dist = np.maximum(sq_dist, 0.0)

    sim = np.exp(-gamma * sq_dist).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def build_heterogeneous_adjacency(
    train_assoc: np.ndarray,
    disease_sim: np.ndarray,
    microbe_sim: np.ndarray,
    sim_threshold: float = 0.3,
) -> np.ndarray:
    """
    构造异构邻接矩阵：
    [ disease-disease   disease-microbe ]
    [ microbe-disease   microbe-microbe ]
    """
    disease_adj = (disease_sim >= sim_threshold).astype(np.float32)
    microbe_adj = (microbe_sim >= sim_threshold).astype(np.float32)

    np.fill_diagonal(disease_adj, 1.0)
    np.fill_diagonal(microbe_adj, 1.0)

    upper = np.hstack((disease_adj, train_assoc))
    lower = np.hstack((train_assoc.T, microbe_adj))
    heter_adj = np.vstack((upper, lower)).astype(np.float32)

    return heter_adj


def _sample_negative_pairs(zero_mask: np.ndarray, num_samples: int, rng) -> np.ndarray:
    candidates = np.argwhere(zero_mask)
    if len(candidates) == 0:
        raise ValueError("No available negative candidates for sampling.")

    num_samples = min(num_samples, len(candidates))
    selected = rng.choice(len(candidates), size=num_samples, replace=False)
    return candidates[selected].astype(np.int64)


def build_cvs_folds(
    assoc_matrix: np.ndarray,
    cvs: str,
    n_splits: int,
    rng,
) -> List[Dict]:
    """
    输出每个 fold 的:
    - train_assoc
    - train_mask
    - test_pairs
    - test_labels
    """
    cvs = cvs.upper()
    folds = []

    if cvs == "CVS1":
        positive_pairs = np.argwhere(assoc_matrix == 1)
        perm = rng.permutation(len(positive_pairs))
        positive_folds = np.array_split(positive_pairs[perm], n_splits)

        for test_pos in positive_folds:
            if len(test_pos) == 0:
                continue

            test_neg = _sample_negative_pairs(
                zero_mask=(assoc_matrix == 0),
                num_samples=len(test_pos),
                rng=rng,
            )

            train_assoc = assoc_matrix.copy()
            train_assoc[test_pos[:, 0], test_pos[:, 1]] = 0

            train_mask = np.ones_like(assoc_matrix, dtype=bool)
            train_mask[test_pos[:, 0], test_pos[:, 1]] = False
            train_mask[test_neg[:, 0], test_neg[:, 1]] = False

            test_pairs = np.vstack((test_pos, test_neg)).astype(np.int64)
            test_labels = np.concatenate(
                (
                    np.ones(len(test_pos), dtype=np.float32),
                    np.zeros(len(test_neg), dtype=np.float32),
                )
            )

            folds.append(
                {
                    "train_assoc": train_assoc,
                    "train_mask": train_mask,
                    "test_pairs": test_pairs,
                    "test_labels": test_labels,
                }
            )

    elif cvs == "CVS2":
        disease_indices = rng.permutation(assoc_matrix.shape[0])
        disease_folds = np.array_split(disease_indices, n_splits)

        for test_rows in disease_folds:
            row_mask = np.zeros_like(assoc_matrix, dtype=bool)
            row_mask[test_rows, :] = True

            test_pos = np.argwhere((assoc_matrix == 1) & row_mask)
            if len(test_pos) == 0:
                continue

            test_neg = _sample_negative_pairs(
                zero_mask=((assoc_matrix == 0) & row_mask),
                num_samples=len(test_pos),
                rng=rng,
            )

            train_assoc = assoc_matrix.copy()
            train_assoc[test_rows, :] = 0

            train_mask = np.ones_like(assoc_matrix, dtype=bool)
            train_mask[test_rows, :] = False

            test_pairs = np.vstack((test_pos, test_neg)).astype(np.int64)
            test_labels = np.concatenate(
                (
                    np.ones(len(test_pos), dtype=np.float32),
                    np.zeros(len(test_neg), dtype=np.float32),
                )
            )

            folds.append(
                {
                    "train_assoc": train_assoc,
                    "train_mask": train_mask,
                    "test_pairs": test_pairs,
                    "test_labels": test_labels,
                }
            )

    elif cvs == "CVS3":
        microbe_indices = rng.permutation(assoc_matrix.shape[1])
        microbe_folds = np.array_split(microbe_indices, n_splits)

        for test_cols in microbe_folds:
            col_mask = np.zeros_like(assoc_matrix, dtype=bool)
            col_mask[:, test_cols] = True

            test_pos = np.argwhere((assoc_matrix == 1) & col_mask)
            if len(test_pos) == 0:
                continue

            test_neg = _sample_negative_pairs(
                zero_mask=((assoc_matrix == 0) & col_mask),
                num_samples=len(test_pos),
                rng=rng,
            )

            train_assoc = assoc_matrix.copy()
            train_assoc[:, test_cols] = 0

            train_mask = np.ones_like(assoc_matrix, dtype=bool)
            train_mask[:, test_cols] = False

            test_pairs = np.vstack((test_pos, test_neg)).astype(np.int64)
            test_labels = np.concatenate(
                (
                    np.ones(len(test_pos), dtype=np.float32),
                    np.zeros(len(test_neg), dtype=np.float32),
                )
            )

            folds.append(
                {
                    "train_assoc": train_assoc,
                    "train_mask": train_mask,
                    "test_pairs": test_pairs,
                    "test_labels": test_labels,
                }
            )

    else:
        raise ValueError(f"Unsupported CVS setting: {cvs}")

    return folds


def sample_train_pairs(
    train_assoc: np.ndarray,
    train_mask: np.ndarray,
    negative_multiplier: float,
    rng,
):
    """
    每个 epoch 动态采样训练对：
    - 正样本：train_assoc 中可见的已知关联
    - 负样本：train_mask 范围内的未知关联
    """
    positive_pairs = np.argwhere((train_assoc == 1) & train_mask)
    if len(positive_pairs) == 0:
        raise ValueError("No positive training pairs found.")

    num_negative = max(1, int(round(len(positive_pairs) * negative_multiplier)))
    negative_pairs = _sample_negative_pairs(
        zero_mask=((train_assoc == 0) & train_mask),
        num_samples=num_negative,
        rng=rng,
    )

    pair_index = np.vstack((positive_pairs, negative_pairs)).astype(np.int64)
    labels = np.concatenate(
        (
            np.ones(len(positive_pairs), dtype=np.float32),
            np.zeros(len(negative_pairs), dtype=np.float32),
        )
    )

    perm = rng.permutation(len(pair_index))
    pair_index = pair_index[perm]
    labels = labels[perm]

    return pair_index, labels


def evaluate_scores(test_labels: np.ndarray, test_scores: np.ndarray) -> Dict:
    """
    主指标：AUC / AUPR
    辅助指标：ACC / REC / PRE / F1
    """
    test_labels = np.asarray(test_labels).astype(int)
    test_scores = np.asarray(test_scores).astype(float)

    auc = roc_auc_score(test_labels, test_scores)
    aupr = average_precision_score(test_labels, test_scores)

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)

    pred_label = (test_scores >= 0.5).astype(int)

    accuracy = accuracy_score(test_labels, pred_label)
    recall = recall_score(test_labels, pred_label, zero_division=0)
    precision = precision_score(test_labels, pred_label, zero_division=0)
    f1 = f1_score(test_labels, pred_label, zero_division=0)

    return {
        "auc": float(auc),
        "aupr": float(aupr),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": precision_curve,
        "pr_recall": recall_curve,
    }

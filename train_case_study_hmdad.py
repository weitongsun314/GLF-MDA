import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 与正式消融实验保持一致：使用 full 版本模型
from model_ablation import Model
from utils import (
    build_heterogeneous_adjacency,
    compute_gip_similarity,
    load_edge_list_matrix,
    sample_train_pairs,
    set_seed,
)


# ============================================================
# 1. 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HMDAD case study: GATMDA-style leave-one-disease-out ranking"
    )

    code_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--new_data_root",
        type=str,
        default=os.path.join(code_dir, "new_data"),
    )
    parser.add_argument(
        "--other_sources_root",
        type=str,
        default=os.path.join(code_dir, "other_sources"),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="HMDAD",
        choices=["HMDAD"],
    )

    # GATMDA 论文案例研究选 Asthma 和 IBD。
    # 在当前 HMDAD diseases.xlsx 中：
    #   4  = Asthma
    #   23 = Inflammatory bowel disease(IBD)
    parser.add_argument(
        "--target_ids",
        type=str,
        default="4,23",
        help="1-based disease IDs, comma separated. Default: Asthma=4, IBD=23",
    )

    # reset_known:
    #   严格按照 GATMDA 案例研究思路：
    #   把目标疾病所有已知关联置为 unknown，并将目标疾病整行排除出训练监督。
    #
    # unknown_only:
    #   使用完整已知网络训练，只对原本未知关联排序。
    #   这是另一种常见案例研究方式，不是当前默认方案。
    parser.add_argument(
        "--mode",
        type=str,
        default="reset_known",
        choices=["reset_known", "unknown_only"],
    )

    # 第一轮先跑 1 次检查排名是否正常。
    # 如果后续希望稳定排名，可设为 10，对多个 seed 的预测分数取均值。
    parser.add_argument(
        "--case_repeats",
        type=int,
        default=10,
    )

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )

    # ========================================================
    # HMDAD 最终参数：与正式实验完全一致
    # ========================================================
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--weight_decay", type=float, default=0.00005)

    parser.add_argument("--feature_out", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--sim_threshold", type=float, default=0.2)

    parser.add_argument("--negative_multiplier", type=float, default=1.0)

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="disease similarity = alpha*GIP + (1-alpha)*DO",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="microbe similarity = beta*GIP + (1-beta)*functional",
    )

    parser.add_argument("--mdmf_dim", type=int, default=128)
    parser.add_argument("--mdmf_loss_weight", type=float, default=0.0)
    parser.add_argument("--mdmf_gate_init", type=float, default=-2.0)

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top candidates saved in the dedicated top-K sheets.",
    )

    return parser.parse_args()


# ============================================================
# 2. 数据工具
# ============================================================

def normalize_similarity_matrix(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    sim = np.nan_to_num(
        sim,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    min_val = float(sim.min())
    max_val = float(sim.max())

    if max_val > min_val:
        sim = (sim - min_val) / (max_val - min_val)

    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def read_similarity_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    return np.nan_to_num(
        df.to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def load_id_name_table(path: str) -> pd.DataFrame:
    """
    diseases.xlsx / microbes.xlsx 均按无表头方式读取：
    column 0 = 1-based ID
    column 1 = name
    """
    df = pd.read_excel(path, header=None)

    if df.shape[1] < 2:
        raise ValueError(f"ID-name table has fewer than two columns: {path}")

    df = df.iloc[:, :2].copy()
    df.columns = ["id", "name"]

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id", "name"]).copy()
    df["id"] = df["id"].astype(int)
    df["name"] = df["name"].astype(str).str.strip()

    return df


def safe_sheet_name(name: str, suffix: str) -> str:
    clean = re.sub(r'[:\\/?*\[\]]', "_", str(name))
    clean = clean.strip()
    full = f"{clean}_{suffix}"
    return full[:31]


# ============================================================
# 3. 案例研究准备
# ============================================================

def prepare_case_data(
    original_assoc: np.ndarray,
    target_idx: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    - train_assoc
    - train_mask
    - candidate_microbe_indices

    reset_known 模式：
    1) 目标疾病整行关联全部清零；
    2) 目标疾病整行 train_mask=False；
       这样该行不会被 sample_train_pairs 当成负样本使用；
    3) 对全部微生物进行候选排序。

    unknown_only 模式：
    1) 使用完整关联矩阵训练；
    2) 仅预测原本未知的微生物。
    """
    train_assoc = original_assoc.copy()
    train_mask = np.ones_like(original_assoc, dtype=bool)

    if mode == "reset_known":
        train_assoc[target_idx, :] = 0.0

        # 极其重要：
        # 目标疾病所有 pair 都不参与训练监督，
        # 避免把 reset 后的未知 pair 当成负样本。
        train_mask[target_idx, :] = False

        candidates = np.arange(
            original_assoc.shape[1],
            dtype=np.int64,
        )

    elif mode == "unknown_only":
        candidates = np.where(
            original_assoc[target_idx, :] == 0
        )[0].astype(np.int64)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return train_assoc, train_mask, candidates


# ============================================================
# 4. 单个疾病、单个 seed 的训练与预测
# ============================================================

def train_and_predict_one_repeat(
    args,
    original_assoc: np.ndarray,
    disease_do_sim: np.ndarray,
    microbe_func_sim: np.ndarray,
    target_idx: int,
    candidate_indices: np.ndarray,
    repeat_idx: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:

    run_seed = args.seed + repeat_idx
    set_seed(run_seed)
    rng = np.random.default_rng(run_seed)

    train_assoc, train_mask, _ = prepare_case_data(
        original_assoc=original_assoc,
        target_idx=target_idx,
        mode=args.mode,
    )

    # --------------------------------------------------------
    # 与正式 HMDAD 训练代码完全一致地构建相似性
    # --------------------------------------------------------
    disease_gip = normalize_similarity_matrix(
        compute_gip_similarity(train_assoc)
    )
    microbe_gip = normalize_similarity_matrix(
        compute_gip_similarity(train_assoc.T)
    )

    disease_sim = normalize_similarity_matrix(
        args.alpha * disease_gip
        + (1.0 - args.alpha) * disease_do_sim
    )

    microbe_sim = normalize_similarity_matrix(
        args.beta * microbe_gip
        + (1.0 - args.beta) * microbe_func_sim
    )

    heter_adj = build_heterogeneous_adjacency(
        train_assoc=train_assoc,
        disease_sim=disease_sim,
        microbe_sim=microbe_sim,
        sim_threshold=args.sim_threshold,
    )

    disease_sim_t = torch.from_numpy(disease_sim).float().to(device)
    microbe_sim_t = torch.from_numpy(microbe_sim).float().to(device)
    heter_adj_t = torch.from_numpy(heter_adj).float().to(device)
    train_assoc_t = torch.from_numpy(train_assoc).float().to(device)

    n_disease, n_microbe = original_assoc.shape

    # 使用正式实验中的 Full 模型
    model = Model(
        disease_feat_in=n_disease,
        microbe_feat_in=n_microbe,
        feature_out=args.feature_out,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        drop_rate=args.drop_rate,
        ablation="full",
        mdmf_dim=args.mdmf_dim,
        mdmf_gate_init=args.mdmf_gate_init,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()

        train_pair_index, train_pair_labels = sample_train_pairs(
            train_assoc=train_assoc,
            train_mask=train_mask,
            negative_multiplier=args.negative_multiplier,
            rng=rng,
        )

        pair_t = torch.from_numpy(
            train_pair_index
        ).long().to(device)

        labels_t = (
            torch.from_numpy(train_pair_labels)
            .float()
            .unsqueeze(1)
            .to(device)
        )

        disease_emb, microbe_emb = model.encode(
            heter_adj=heter_adj_t,
            disease_feature=disease_sim_t,
            microbe_feature=microbe_sim_t,
            train_assoc=train_assoc_t,
        )

        pred = model.predict_pairs(
            disease_emb,
            microbe_emb,
            pair_t,
        )

        main_loss = F.binary_cross_entropy(
            pred,
            labels_t,
        )

        loss = main_loss

        if model.use_mdmf and args.mdmf_loss_weight > 0:
            mdmf_pred = model.mdmf_predict_pairs(
                train_assoc=train_assoc_t,
                disease_feature=disease_sim_t,
                microbe_feature=microbe_sim_t,
                pair_index=pair_t,
            )

            mdmf_loss = F.binary_cross_entropy(
                mdmf_pred,
                labels_t,
            )

            loss = (
                loss
                + args.mdmf_loss_weight * mdmf_loss
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
            or epoch + 1 == args.epochs
        ):
            print(
                f"    seed={run_seed} "
                f"epoch={epoch + 1:03d}/{args.epochs} "
                f"loss={loss.item():.6f}"
            )

    # --------------------------------------------------------
    # 对目标疾病候选微生物打分
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():
        disease_emb, microbe_emb = model.encode(
            heter_adj=heter_adj_t,
            disease_feature=disease_sim_t,
            microbe_feature=microbe_sim_t,
            train_assoc=train_assoc_t,
        )

        pair_index = np.column_stack(
            (
                np.full(
                    len(candidate_indices),
                    target_idx,
                    dtype=np.int64,
                ),
                candidate_indices.astype(np.int64),
            )
        )

        pair_t = torch.from_numpy(
            pair_index
        ).long().to(device)

        scores = (
            model.predict_pairs(
                disease_emb,
                microbe_emb,
                pair_t,
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        gate_value = (
            torch.sigmoid(model.mdmf_gate)
            .detach()
            .cpu()
            .item()
            if model.use_mdmf
            else 0.0
        )

    return scores.astype(float), float(gate_value)


# ============================================================
# 5. 单个疾病完整案例研究
# ============================================================

def run_case_for_disease(
    args,
    original_assoc: np.ndarray,
    disease_do_sim: np.ndarray,
    microbe_func_sim: np.ndarray,
    disease_row: pd.Series,
    microbes_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict]:

    disease_id = int(disease_row["id"])
    disease_name = str(disease_row["name"])
    target_idx = disease_id - 1

    if not (0 <= target_idx < original_assoc.shape[0]):
        raise ValueError(
            f"Disease ID {disease_id} is outside association matrix."
        )

    original_known_indices = np.where(
        original_assoc[target_idx, :] == 1
    )[0]

    _, _, candidate_indices = prepare_case_data(
        original_assoc=original_assoc,
        target_idx=target_idx,
        mode=args.mode,
    )

    print("\n" + "=" * 90)
    print(
        f"Case study: {disease_name} "
        f"(ID={disease_id}, matrix index={target_idx})"
    )
    print(f"Mode: {args.mode}")
    print(
        f"Original known associations: "
        f"{len(original_known_indices)}"
    )
    print(
        f"Candidate microbes to rank: "
        f"{len(candidate_indices)}"
    )
    print("=" * 90)

    score_runs = []
    gate_runs = []

    for repeat_idx in range(args.case_repeats):
        print(
            f"\n  Repeat {repeat_idx + 1}/"
            f"{args.case_repeats}"
        )

        scores, gate = train_and_predict_one_repeat(
            args=args,
            original_assoc=original_assoc,
            disease_do_sim=disease_do_sim,
            microbe_func_sim=microbe_func_sim,
            target_idx=target_idx,
            candidate_indices=candidate_indices,
            repeat_idx=repeat_idx,
            device=device,
        )

        score_runs.append(scores)
        gate_runs.append(gate)

    score_matrix = np.vstack(score_runs)

    score_mean = score_matrix.mean(axis=0)
    score_std = (
        score_matrix.std(axis=0, ddof=1)
        if args.case_repeats > 1
        else np.zeros_like(score_mean)
    )

    microbe_name_map = dict(
        zip(
            microbes_df["id"].astype(int),
            microbes_df["name"].astype(str),
        )
    )

    result_df = pd.DataFrame(
        {
            "microbe_index_0based": candidate_indices,
            "microbe_id": candidate_indices + 1,
            "microbe_name": [
                microbe_name_map.get(
                    int(idx + 1),
                    f"Microbe_{idx + 1}",
                )
                for idx in candidate_indices
            ],
            "score_mean": score_mean,
            "score_std": score_std,
            "originally_known_in_HMDAD": [
                int(
                    original_assoc[
                        target_idx,
                        int(idx),
                    ]
                    == 1
                )
                for idx in candidate_indices
            ],
        }
    )

    result_df = (
        result_df
        .sort_values(
            "score_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result_df) + 1,
        ),
    )

    # 给后续人工/文献检索留好列
    result_df["literature_status"] = ""
    result_df["PMID_or_DOI"] = ""
    result_df["evidence_notes"] = ""

    summary = {
        "disease_id": disease_id,
        "disease_name": disease_name,
        "mode": args.mode,
        "case_repeats": args.case_repeats,
        "original_known_count": int(
            len(original_known_indices)
        ),
        "candidate_count": int(
            len(candidate_indices)
        ),
        "top10_original_known_recovered": int(
            result_df.head(10)[
                "originally_known_in_HMDAD"
            ].sum()
        ),
        "top20_original_known_recovered": int(
            result_df.head(20)[
                "originally_known_in_HMDAD"
            ].sum()
        ),
        "top50_original_known_recovered": int(
            result_df.head(50)[
                "originally_known_in_HMDAD"
            ].sum()
        ),
        "mdmf_gate_mean": float(
            np.mean(gate_runs)
        ),
        "mdmf_gate_std": float(
            np.std(gate_runs, ddof=1)
            if len(gate_runs) > 1
            else 0.0
        ),
    }

    return result_df, summary


# ============================================================
# 6. 主程序
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset_dir = os.path.join(
        args.new_data_root,
        args.dataset,
    )

    adj_path = os.path.join(
        dataset_dir,
        "adj.txt",
    )
    diseases_path = os.path.join(
        dataset_dir,
        "diseases.xlsx",
    )
    microbes_path = os.path.join(
        dataset_dir,
        "microbes.xlsx",
    )

    disease_do_path = os.path.join(
        args.other_sources_root,
        "disease_do_similarity.csv",
    )
    microbe_func_path = os.path.join(
        args.other_sources_root,
        "microbe_functional_similarity.csv",
    )

    required_paths = [
        adj_path,
        diseases_path,
        microbes_path,
        disease_do_path,
        microbe_func_path,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    original_assoc = load_edge_list_matrix(
        adj_path,
        one_based=True,
    ).astype(np.float32)

    n_disease, n_microbe = (
        original_assoc.shape
    )

    diseases_df = load_id_name_table(
        diseases_path
    )
    microbes_df = load_id_name_table(
        microbes_path
    )

    disease_do_sim = normalize_similarity_matrix(
        read_similarity_csv(
            disease_do_path
        )
    )
    microbe_func_sim = normalize_similarity_matrix(
        read_similarity_csv(
            microbe_func_path
        )
    )

    if disease_do_sim.shape != (
        n_disease,
        n_disease,
    ):
        raise ValueError(
            "Disease similarity shape mismatch: "
            f"{disease_do_sim.shape} vs "
            f"{(n_disease, n_disease)}"
        )

    if microbe_func_sim.shape != (
        n_microbe,
        n_microbe,
    ):
        raise ValueError(
            "Microbe similarity shape mismatch: "
            f"{microbe_func_sim.shape} vs "
            f"{(n_microbe, n_microbe)}"
        )

    target_ids = [
        int(x.strip())
        for x in args.target_ids.split(",")
        if x.strip()
    ]

    disease_lookup = (
        diseases_df
        .set_index("id")
    )

    missing_ids = [
        x
        for x in target_ids
        if x not in disease_lookup.index
    ]

    if missing_ids:
        raise ValueError(
            f"Disease IDs not found: {missing_ids}"
        )

    print("\n" + "#" * 90)
    print("HMDAD CASE STUDY")
    print("#" * 90)
    print(f"Association shape: {original_assoc.shape}")
    print(f"Known associations: {int(original_assoc.sum())}")
    print(f"Mode: {args.mode}")
    print(f"Targets: {target_ids}")
    print(f"Case repeats: {args.case_repeats}")
    print(
        "HMDAD final params: "
        f"epochs={args.epochs}, "
        f"lr={args.lr}, "
        f"wd={args.weight_decay}, "
        f"hidden={args.hidden_size}, "
        f"dropout={args.dropout}, "
        f"drop_rate={args.drop_rate}, "
        f"threshold={args.sim_threshold}, "
        f"alpha={args.alpha}, "
        f"beta={args.beta}"
    )
    print("#" * 90)

    all_results: Dict[int, pd.DataFrame] = {}
    summaries: List[Dict] = []

    for disease_id in target_ids:
        disease_row = disease_lookup.loc[
            disease_id
        ]

        # disease_lookup.loc 返回 Series，
        # 把 id 补回来方便下游函数使用
        disease_row = disease_row.copy()
        disease_row["id"] = disease_id

        result_df, summary = run_case_for_disease(
            args=args,
            original_assoc=original_assoc,
            disease_do_sim=disease_do_sim,
            microbe_func_sim=microbe_func_sim,
            disease_row=disease_row,
            microbes_df=microbes_df,
            device=device,
        )

        all_results[disease_id] = result_df
        summaries.append(summary)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    results_dir = os.path.join(
        args.other_sources_root,
        "results",
        "case_study",
    )
    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        results_dir,
        (
            f"HMDAD_case_study_{args.mode}"
            f"_repeats{args.case_repeats}"
            f"_seed{args.seed}.xlsx"
        ),
    )

    config = {
        "dataset": args.dataset,
        "case_study_mode": args.mode,
        "target_disease_ids": args.target_ids,
        "case_repeats": args.case_repeats,
        "seed_start": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "feature_out": args.feature_out,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "drop_rate": args.drop_rate,
        "sim_threshold": args.sim_threshold,
        "negative_multiplier": args.negative_multiplier,
        "alpha": args.alpha,
        "beta": args.beta,
        "mdmf_dim": args.mdmf_dim,
        "mdmf_loss_weight": args.mdmf_loss_weight,
        "mdmf_gate_init": args.mdmf_gate_init,
        "model_variant": "full",
        "ranking_score": (
            "mean predicted probability across "
            f"{args.case_repeats} run(s)"
        ),
        "reset_known_training_rule": (
            "Target disease row is set to zero and "
            "excluded from positive/negative training sampling."
            if args.mode == "reset_known"
            else "Full known network is used for training."
        ),
    }

    with pd.ExcelWriter(
        save_path,
        engine="openpyxl",
    ) as writer:

        pd.DataFrame(
            summaries
        ).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        pd.DataFrame(
            [config]
        ).to_excel(
            writer,
            sheet_name="config",
            index=False,
        )

        for disease_id in target_ids:
            result_df = all_results[
                disease_id
            ]

            disease_name = str(
                disease_lookup.loc[
                    disease_id,
                    "name",
                ]
            )

            # 使用短名称，便于 Excel sheet
            if disease_name.lower() == "asthma":
                short_name = "Asthma"
            elif "inflammatory bowel disease" in disease_name.lower():
                short_name = "IBD"
            else:
                short_name = f"D{disease_id}"

            result_df.head(
                args.top_k
            ).to_excel(
                writer,
                sheet_name=safe_sheet_name(
                    short_name,
                    f"top{args.top_k}",
                ),
                index=False,
            )

            result_df.to_excel(
                writer,
                sheet_name=safe_sheet_name(
                    short_name,
                    "all",
                ),
                index=False,
            )

    print("\n" + "#" * 90)
    print("CASE STUDY FINISHED")
    print(f"Saved to: {save_path}")
    print("#" * 90)

    print("\nSummary:")
    print(
        pd.DataFrame(
            summaries
        ).to_string(index=False)
    )

    print(
        "\n注意：top10/top20/top50_original_known_recovered "
        "只是检查模型是否重新找回 HMDAD 中原有已知关联，"
        "不是最终文献验证率。最终 case study 仍需对 Top 50 "
        "逐条检索文献/PMID。"
    )


if __name__ == "__main__":
    main()

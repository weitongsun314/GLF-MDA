import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_mdmf import Model
from utils import (
    build_cvs_folds,
    build_heterogeneous_adjacency,
    compute_gip_similarity,
    evaluate_scores,
    load_edge_list_matrix,
    sample_train_pairs,
    set_seed,
)


def get_default_dropout(dataset_name: str) -> float:
    dataset_name = dataset_name.upper()
    if dataset_name == "HMDAD":
        return 0.5
    if dataset_name == "DISBIOME":
        return 0.3
    return 0.3


def parse_args():
    parser = argparse.ArgumentParser()

    # Mac/Windows 通用路径：
    # 以 train_mdmf.py 所在文件夹作为项目根目录。
    # 只要你的项目结构保持为：
    # 科研/代码/train_mdmf.py
    # 科研/代码/new_data/
    # 科研/代码/other_sources/
    # 这段代码就会自动找到数据集和相似性矩阵，不再绑定 Windows 绝对路径。
    code_dir = os.path.dirname(os.path.abspath(__file__))
    default_new_data_root = os.path.join(code_dir, "new_data")
    default_other_sources_root = os.path.join(code_dir, "other_sources")

    parser.add_argument(
        "--new_data_root",
        type=str,
        default=default_new_data_root
    )
    parser.add_argument(
        "--other_sources_root",
        type=str,
        default=default_other_sources_root
    )

    # 单独运行时的默认设置
    parser.add_argument("--dataset", type=str, default="HMDAD", choices=["HMDAD", "Disbiome"])
    parser.add_argument("--cvs", type=str, default="CVS1", choices=["CVS1", "CVS2", "CVS3"])

    # 默认批量跑：HMDAD 和 Disbiome 的 CVS1
    # 这样运行 python train_mdmf.py 时，会依次生成：
    # HMDAD_CVS1_MDMF_results.xlsx
    # Disbiome_CVS1_MDMF_results.xlsx
    parser.add_argument("--run_all", action="store_true", default=True)
    parser.add_argument("--datasets", type=str, default="HMDAD,Disbiome")
    parser.add_argument("--cvs_list", type=str, default="CVS1")

    parser.add_argument("--folds", type=int, default=5)

    # 正式跑用 100。
    # 如果你只是想测试 Mac 路径和 Excel 保存是否正常，可以临时改成 1。
    parser.add_argument("--repeats", type=int, default=10)

    parser.add_argument("--seed", type=int, default=2025)

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

    # 苹果电脑先用 CPU，最稳。
    # 这里保留 cuda 选项，是为了后面如果换回 Windows/NVIDIA GPU 也能继续用。
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--alpha", type=float, default=1.0, help="disease: alpha*GIP + (1-alpha)*DO")
    parser.add_argument("--beta", type=float, default=0.25, help="microbe: beta*GIP + (1-beta)*functional")

    # MDMF 分支超参数：
    # use_mdmf=1：打开 MDMF 分支，也就是“原模型 + MDMF”。
    # use_mdmf=0：关闭 MDMF 分支，可用于后续消融实验。
    parser.add_argument("--use_mdmf", type=int, default=1, help="1=使用 MDMF 分支；0=关闭 MDMF 分支")
    parser.add_argument("--mdmf_dim", type=int, default=128, help="MDMF 低秩隐空间维度")
    parser.add_argument("--mdmf_loss_weight", type=float, default=0, help="MDMF 辅助重构 loss 权重")
    parser.add_argument("--mdmf_gate_init", type=float, default=-2.0, help="MDMF gate 初始 logit；sigmoid(-2)≈0.12")

    return parser.parse_args()


def normalize_similarity_matrix(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    min_val = sim.min()
    max_val = sim.max()

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

    sim = df.to_numpy(dtype=np.float32)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim


def get_external_similarity_paths(other_sources_root: str, dataset_name: str):
    dataset_upper = dataset_name.upper()

    if dataset_upper == "DISBIOME":
        disease_do_path = os.path.join(other_sources_root, "disease_do_similarity Disbiome.csv")
        microbe_func_path = os.path.join(other_sources_root, "microbe_functional_similarity Disbiome.csv")
    else:
        disease_do_path = os.path.join(other_sources_root, "disease_do_similarity.csv")
        microbe_func_path = os.path.join(other_sources_root, "microbe_functional_similarity.csv")

    return disease_do_path, microbe_func_path


def check_matrix_shape(matrix: np.ndarray, expected_shape: tuple, matrix_name: str):
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{matrix_name} shape mismatch: got {matrix.shape}, expected {expected_shape}. "
            f"请检查该相似性矩阵是否和当前数据集对应。"
        )


def run_one_config(args, dataset_name: str, cvs_name: str, device: torch.device):
    set_seed(args.seed)

    dropout = args.dropout
    if dropout is None:
        dropout = get_default_dropout(dataset_name)

    adj_path = os.path.join(args.new_data_root, dataset_name, "adj.txt")
    assoc_matrix = load_edge_list_matrix(adj_path, one_based=True).astype(np.float32)

    n_disease, n_microbe = assoc_matrix.shape

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"CV setting: {cvs_name}")
    print(f"adj path: {adj_path}")
    print(f"Association matrix shape: {assoc_matrix.shape}")
    print(f"Positives: {int(assoc_matrix.sum())}")
    print("=" * 70)

    disease_do_path, microbe_func_path = get_external_similarity_paths(
        args.other_sources_root, dataset_name
    )

    print(f"Disease similarity path: {disease_do_path}")
    print(f"Microbe similarity path: {microbe_func_path}")

    disease_do_sim = read_similarity_csv(disease_do_path)
    microbe_func_sim = read_similarity_csv(microbe_func_path)

    check_matrix_shape(disease_do_sim, (n_disease, n_disease), "disease_do_sim")
    check_matrix_shape(microbe_func_sim, (n_microbe, n_microbe), "microbe_func_sim")

    disease_do_sim = normalize_similarity_matrix(disease_do_sim)
    microbe_func_sim = normalize_similarity_matrix(microbe_func_sim)

    all_fold_results = []

    for repeat_idx in range(args.repeats):
        print(f"\n========== Repeat {repeat_idx + 1}/{args.repeats} ==========")
        rng = np.random.default_rng(args.seed + repeat_idx)

        folds = build_cvs_folds(
            assoc_matrix=assoc_matrix,
            cvs=cvs_name,
            n_splits=args.folds,
            rng=rng,
        )

        for fold_idx, fold_data in enumerate(folds, start=1):
            print(f"\n----- Fold {fold_idx}/{len(folds)} -----")

            train_assoc = fold_data["train_assoc"]
            train_mask = fold_data["train_mask"]
            test_pairs = fold_data["test_pairs"]
            test_labels = fold_data["test_labels"]

            disease_gip = compute_gip_similarity(train_assoc)
            microbe_gip = compute_gip_similarity(train_assoc.T)

            disease_gip = normalize_similarity_matrix(disease_gip)
            microbe_gip = normalize_similarity_matrix(microbe_gip)

            if disease_gip.shape != disease_do_sim.shape:
                raise ValueError(
                    f"disease shape mismatch: disease_gip={disease_gip.shape}, "
                    f"disease_do_sim={disease_do_sim.shape}"
                )

            if microbe_gip.shape != microbe_func_sim.shape:
                raise ValueError(
                    f"microbe shape mismatch: microbe_gip={microbe_gip.shape}, "
                    f"microbe_func_sim={microbe_func_sim.shape}"
                )

            disease_sim_final = args.alpha * disease_gip + (1.0 - args.alpha) * disease_do_sim
            microbe_sim_final = args.beta * microbe_gip + (1.0 - args.beta) * microbe_func_sim

            disease_sim_final = normalize_similarity_matrix(disease_sim_final)
            microbe_sim_final = normalize_similarity_matrix(microbe_sim_final)

            heter_adj = build_heterogeneous_adjacency(
                train_assoc=train_assoc,
                disease_sim=disease_sim_final,
                microbe_sim=microbe_sim_final,
                sim_threshold=args.sim_threshold,
            )

            disease_sim_t = torch.from_numpy(disease_sim_final).float().to(device)
            microbe_sim_t = torch.from_numpy(microbe_sim_final).float().to(device)
            heter_adj_t = torch.from_numpy(heter_adj).float().to(device)
            train_assoc_t = torch.from_numpy(train_assoc).float().to(device)

            model = Model(
                disease_feat_in=n_disease,
                microbe_feat_in=n_microbe,
                feature_out=args.feature_out,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=dropout,
                drop_rate=args.drop_rate,
                use_mdmf=bool(args.use_mdmf),
                mdmf_dim=args.mdmf_dim,
                mdmf_gate_init=args.mdmf_gate_init,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            for epoch in range(args.epochs):
                model.train()

                train_pair_index, train_pair_labels = sample_train_pairs(
                    train_assoc=train_assoc,
                    train_mask=train_mask,
                    negative_multiplier=args.negative_multiplier,
                    rng=rng,
                )

                train_pair_index_t = torch.from_numpy(train_pair_index).long().to(device)
                train_pair_labels_t = (
                    torch.from_numpy(train_pair_labels).float().unsqueeze(1).to(device)
                )

                disease_emb, microbe_emb = model.encode(
                    heter_adj=heter_adj_t,
                    disease_feature=disease_sim_t,
                    microbe_feature=microbe_sim_t,
                    train_assoc=train_assoc_t,
                )

                train_pred = model.predict_pairs(
                    disease_embeddings=disease_emb,
                    microbe_embeddings=microbe_emb,
                    pair_index=train_pair_index_t,
                )

                main_loss = F.binary_cross_entropy(train_pred, train_pair_labels_t)
                loss = main_loss
                mdmf_loss = None

                if bool(args.use_mdmf) and args.mdmf_loss_weight > 0:
                    mdmf_pred = model.mdmf_predict_pairs(
                        train_assoc=train_assoc_t,
                        disease_feature=disease_sim_t,
                        microbe_feature=microbe_sim_t,
                        pair_index=train_pair_index_t,
                    )
                    mdmf_loss = F.binary_cross_entropy(mdmf_pred, train_pair_labels_t)
                    loss = loss + args.mdmf_loss_weight * mdmf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    if mdmf_loss is None:
                        print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.6f} | Main: {main_loss.item():.6f}")
                    else:
                        gate_value = torch.sigmoid(model.mdmf_gate).detach().cpu().item()
                        print(
                            f"Epoch {epoch + 1:03d} | Loss: {loss.item():.6f} "
                            f"| Main: {main_loss.item():.6f} "
                            f"| MDMF: {mdmf_loss.item():.6f} "
                            f"| gate: {gate_value:.4f}"
                        )

            model.eval()
            with torch.no_grad():
                disease_emb, microbe_emb = model.encode(
                    heter_adj=heter_adj_t,
                    disease_feature=disease_sim_t,
                    microbe_feature=microbe_sim_t,
                    train_assoc=train_assoc_t,
                )

                test_pairs_t = torch.from_numpy(test_pairs).long().to(device)
                test_scores = model.predict_pairs(
                    disease_embeddings=disease_emb,
                    microbe_embeddings=microbe_emb,
                    pair_index=test_pairs_t,
                ).squeeze(1).cpu().numpy()

            fold_result = evaluate_scores(test_labels, test_scores)
            fold_result["repeat"] = repeat_idx + 1
            fold_result["fold"] = fold_idx
            fold_result["dataset"] = dataset_name
            fold_result["cvs"] = cvs_name
            fold_result["alpha"] = args.alpha
            fold_result["beta"] = args.beta
            fold_result["use_mdmf"] = int(bool(args.use_mdmf))
            fold_result["mdmf_dim"] = args.mdmf_dim
            fold_result["mdmf_loss_weight"] = args.mdmf_loss_weight
            if bool(args.use_mdmf):
                fold_result["mdmf_gate"] = torch.sigmoid(model.mdmf_gate).detach().cpu().item()
            else:
                fold_result["mdmf_gate"] = 0.0

            all_fold_results.append(fold_result)

    results_dir = os.path.join(args.other_sources_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(all_fold_results)

    scalar_metric_cols = ["auc", "aupr", "accuracy", "recall", "precision", "f1"]

    mean_dict = {
        "repeat": "mean",
        "fold": "mean",
        "dataset": dataset_name,
        "cvs": cvs_name,
        "alpha": args.alpha,
        "beta": args.beta,
        "use_mdmf": int(bool(args.use_mdmf)),
        "mdmf_dim": args.mdmf_dim,
        "mdmf_loss_weight": args.mdmf_loss_weight,
    }

    for col in scalar_metric_cols:
        if col in df.columns:
            mean_dict[col] = df[col].astype(float).mean()

    if "mdmf_gate" in df.columns:
        mean_dict["mdmf_gate"] = df["mdmf_gate"].astype(float).mean()

    df_mean = pd.DataFrame([mean_dict])

    save_path = os.path.join(results_dir, f"{dataset_name}_{cvs_name}_MDMF_results.xlsx")
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)
        df_mean.to_excel(writer, sheet_name="mean", index=False)

    print(f"\nSummary saved to: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if args.run_all:
        for dataset_name in [x.strip() for x in args.datasets.split(",")]:
            for cvs_name in [x.strip() for x in args.cvs_list.split(",")]:
                run_one_config(args, dataset_name, cvs_name, device)
    else:
        run_one_config(args, args.dataset, args.cvs, device)

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_ablation import Model
from utils import (
    build_cvs_folds,
    build_heterogeneous_adjacency,
    compute_gip_similarity,
    evaluate_scores,
    load_edge_list_matrix,
    sample_train_pairs,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Microbe-disease ablation experiments")

    code_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--new_data_root", type=str, default=os.path.join(code_dir, "new_data"))
    parser.add_argument("--other_sources_root", type=str, default=os.path.join(code_dir, "other_sources"))

    # 默认同时跑两个最终数据集；只跑一个时可传 --datasets HMDAD
    parser.add_argument("--datasets", type=str, default="HMDAD,Disbiome")
    parser.add_argument("--cvs", type=str, default="CVS1", choices=["CVS1", "CVS2", "CVS3"])

    # 第一轮建议只跑 no_mdmf；以后可换 no_cnn / gcn。
    parser.add_argument(
        "--ablation",
        type=str,
        default="gcn",
        choices=["full", "no_mdmf", "no_cnn", "gcn"],
    )

    # 最终实验协议
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # 唯一的数据集差异：HMDAD 300 epochs，Disbiome 100 epochs
    parser.add_argument("--hmdad_epochs", type=int, default=300)
    parser.add_argument("--disbiome_epochs", type=int, default=100)

    # 兼容保留：真正的最终参数由 get_dataset_params() 按数据集分别指定
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--weight_decay", type=float, default=0.00005)
    parser.add_argument("--feature_out", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--sim_threshold", type=float, default=0.2)
    parser.add_argument("--negative_multiplier", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--mdmf_dim", type=int, default=128)
    parser.add_argument("--mdmf_loss_weight", type=float, default=0.0)
    parser.add_argument("--mdmf_gate_init", type=float, default=-2.0)

    return parser.parse_args()


def normalize_similarity_matrix(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = sim.min(), sim.max()
    if max_val > min_val:
        sim = (sim - min_val) / (max_val - min_val)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def read_similarity_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return np.nan_to_num(df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def get_external_similarity_paths(root: str, dataset_name: str):
    if dataset_name.upper() == "DISBIOME":
        return (
            os.path.join(root, "disease_do_similarity Disbiome.csv"),
            os.path.join(root, "microbe_functional_similarity Disbiome.csv"),
        )
    return (
        os.path.join(root, "disease_do_similarity.csv"),
        os.path.join(root, "microbe_functional_similarity.csv"),
    )


def get_epochs(args, dataset_name: str) -> int:
    return args.hmdad_epochs if dataset_name.upper() == "HMDAD" else args.disbiome_epochs


def get_dataset_params(args, dataset_name: str) -> dict:
    """
    两个数据集分别使用各自已经确定的最终参数。
    正式消融时，同一个数据集的 full / no_mdmf / no_cnn / gcn
    必须使用完全相同的这组参数。
    """
    name = dataset_name.upper()

    if name == "HMDAD":
        return {
            "epochs": 300,
            "lr": 0.00025,
            "weight_decay": 0.00005,
            "feature_out": 256,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.5,
            "drop_rate": 0.1,
            "sim_threshold": 0.2,
            "negative_multiplier": 1.0,
            "alpha": 1.0,
            "beta": 0.25,
            "mdmf_dim": 128,
            "mdmf_loss_weight": 0.0,
            "mdmf_gate_init": -2.0,
        }

    if name == "DISBIOME":
        return {
            "epochs": 100,
            "lr": 0.00025,
            "weight_decay": 0.00005,
            "feature_out": 256,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "drop_rate": 0.3,
            "sim_threshold": 0.1,
            "negative_multiplier": 1.0,
            "alpha": 1.0,
            "beta": 0.25,
            "mdmf_dim": 128,
            "mdmf_loss_weight": 0.0,
            "mdmf_gate_init": -2.0,
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")


def run_one(args, dataset_name: str, device: torch.device):
    set_seed(args.seed)
    params = get_dataset_params(args, dataset_name)
    epochs = params["epochs"]

    adj_path = os.path.join(args.new_data_root, dataset_name, "adj.txt")
    assoc_matrix = load_edge_list_matrix(adj_path, one_based=True).astype(np.float32)
    n_disease, n_microbe = assoc_matrix.shape

    disease_do_path, microbe_func_path = get_external_similarity_paths(
        args.other_sources_root, dataset_name
    )
    disease_do_sim = normalize_similarity_matrix(read_similarity_csv(disease_do_path))
    microbe_func_sim = normalize_similarity_matrix(read_similarity_csv(microbe_func_path))

    if disease_do_sim.shape != (n_disease, n_disease):
        raise ValueError(f"disease similarity shape mismatch: {disease_do_sim.shape}")
    if microbe_func_sim.shape != (n_microbe, n_microbe):
        raise ValueError(f"microbe similarity shape mismatch: {microbe_func_sim.shape}")

    print("\n" + "=" * 80)
    print(f"Dataset={dataset_name} | Ablation={args.ablation} | CVS={args.cvs}")
    print(f"folds={args.folds} | repeats={args.repeats} | epochs={epochs}")
    print(
        f"lr={params['lr']} | weight_decay={params['weight_decay']} | "
        f"hidden_size={params['hidden_size']} | dropout={params['dropout']} | "
        f"drop_rate={params['drop_rate']} | sim_threshold={params['sim_threshold']}"
    )
    print("=" * 80)

    all_fold_results = []

    for repeat_idx in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat_idx)
        folds = build_cvs_folds(
            assoc_matrix=assoc_matrix,
            cvs=args.cvs,
            n_splits=args.folds,
            rng=rng,
        )

        for fold_idx, fold_data in enumerate(folds, start=1):
            print(f"[{dataset_name}][{args.ablation}] repeat {repeat_idx+1}/{args.repeats}, fold {fold_idx}/{args.folds}")

            train_assoc = fold_data["train_assoc"]
            train_mask = fold_data["train_mask"]
            test_pairs = fold_data["test_pairs"]
            test_labels = fold_data["test_labels"]

            disease_gip = normalize_similarity_matrix(compute_gip_similarity(train_assoc))
            microbe_gip = normalize_similarity_matrix(compute_gip_similarity(train_assoc.T))

            disease_sim = normalize_similarity_matrix(
                params["alpha"] * disease_gip + (1.0 - params["alpha"]) * disease_do_sim
            )
            microbe_sim = normalize_similarity_matrix(
                params["beta"] * microbe_gip + (1.0 - params["beta"]) * microbe_func_sim
            )

            heter_adj = build_heterogeneous_adjacency(
                train_assoc=train_assoc,
                disease_sim=disease_sim,
                microbe_sim=microbe_sim,
                sim_threshold=params["sim_threshold"],
            )

            disease_sim_t = torch.from_numpy(disease_sim).float().to(device)
            microbe_sim_t = torch.from_numpy(microbe_sim).float().to(device)
            heter_adj_t = torch.from_numpy(heter_adj).float().to(device)
            train_assoc_t = torch.from_numpy(train_assoc).float().to(device)

            model = Model(
                disease_feat_in=n_disease,
                microbe_feat_in=n_microbe,
                feature_out=params["feature_out"],
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                drop_rate=params["drop_rate"],
                ablation=args.ablation,
                mdmf_dim=params["mdmf_dim"],
                mdmf_gate_init=params["mdmf_gate_init"],
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )

            for epoch in range(epochs):
                model.train()
                train_pair_index, train_pair_labels = sample_train_pairs(
                    train_assoc=train_assoc,
                    train_mask=train_mask,
                    negative_multiplier=params["negative_multiplier"],
                    rng=rng,
                )
                pair_t = torch.from_numpy(train_pair_index).long().to(device)
                labels_t = torch.from_numpy(train_pair_labels).float().unsqueeze(1).to(device)

                disease_emb, microbe_emb = model.encode(
                    heter_adj=heter_adj_t,
                    disease_feature=disease_sim_t,
                    microbe_feature=microbe_sim_t,
                    train_assoc=train_assoc_t,
                )
                pred = model.predict_pairs(disease_emb, microbe_emb, pair_t)
                main_loss = F.binary_cross_entropy(pred, labels_t)
                loss = main_loss

                # 当前最终配置 mdmf_loss_weight=0，所以默认不会进入。
                # 保留此逻辑只是为了与最终训练代码结构一致。
                if model.use_mdmf and params["mdmf_loss_weight"] > 0:
                    mdmf_pred = model.mdmf_predict_pairs(
                        train_assoc=train_assoc_t,
                        disease_feature=disease_sim_t,
                        microbe_feature=microbe_sim_t,
                        pair_index=pair_t,
                    )
                    loss = loss + params["mdmf_loss_weight"] * F.binary_cross_entropy(mdmf_pred, labels_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                    disease_emb, microbe_emb, test_pairs_t
                ).squeeze(1).cpu().numpy()

            result = evaluate_scores(test_labels, test_scores)
            result.update({
                "repeat": repeat_idx + 1,
                "fold": fold_idx,
                "dataset": dataset_name,
                "cvs": args.cvs,
                "ablation": args.ablation,
                "epochs": epochs,
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "feature_out": params["feature_out"],
                "hidden_size": params["hidden_size"],
                "num_layers": params["num_layers"],
                "dropout": params["dropout"],
                "drop_rate": params["drop_rate"],
                "sim_threshold": params["sim_threshold"],
                "negative_multiplier": params["negative_multiplier"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "use_scalegcn": int(model.use_scalegcn),
                "use_cnn": int(model.use_cnn),
                "use_mdmf": int(model.use_mdmf),
                "mdmf_dim": params["mdmf_dim"],
                "mdmf_loss_weight": params["mdmf_loss_weight"],
                "mdmf_gate": (
                    torch.sigmoid(model.mdmf_gate).detach().cpu().item()
                    if model.use_mdmf else 0.0
                ),
            })
            all_fold_results.append(result)

    df = pd.DataFrame(all_fold_results)
    metric_cols = ["auc", "aupr", "accuracy", "recall", "precision", "f1"]

    mean_row = {
        "repeat": "mean",
        "fold": "mean",
        "dataset": dataset_name,
        "cvs": args.cvs,
        "ablation": args.ablation,
        "epochs": epochs,
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "feature_out": params["feature_out"],
        "hidden_size": params["hidden_size"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "drop_rate": params["drop_rate"],
        "sim_threshold": params["sim_threshold"],
        "negative_multiplier": params["negative_multiplier"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "use_scalegcn": int(args.ablation != "gcn"),
        "use_cnn": int(args.ablation != "no_cnn"),
        "use_mdmf": int(args.ablation != "no_mdmf"),
        "mdmf_dim": params["mdmf_dim"],
        "mdmf_loss_weight": params["mdmf_loss_weight"],
    }
    for col in metric_cols:
        mean_row[col] = df[col].astype(float).mean()
    mean_row["mdmf_gate"] = df["mdmf_gate"].astype(float).mean()

    # 单独保存每个消融版本，绝不覆盖原来的最终结果文件。
    results_dir = os.path.join(args.other_sources_root, "results", "ablation")
    os.makedirs(results_dir, exist_ok=True)
    param_tag = (
        f"hs{params['hidden_size']}"
        f"_do{params['dropout']:g}"
        f"_dr{params['drop_rate']:g}"
        f"_st{params['sim_threshold']:g}"
    )
    save_path = os.path.join(
        results_dir,
        f"{dataset_name}_{args.cvs}_{args.ablation}_{param_tag}_results.xlsx"
    )

    config = {
        "dataset": dataset_name,
        "cvs": args.cvs,
        "ablation": args.ablation,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "epochs": epochs,
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "feature_out": params["feature_out"],
        "hidden_size": params["hidden_size"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "drop_rate": params["drop_rate"],
        "sim_threshold": params["sim_threshold"],
        "negative_multiplier": params["negative_multiplier"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "mdmf_dim": params["mdmf_dim"],
        "mdmf_loss_weight": params["mdmf_loss_weight"],
        "mdmf_gate_init": params["mdmf_gate_init"],
    }

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)
        pd.DataFrame([mean_row]).to_excel(writer, sheet_name="mean", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="config", index=False)

    print("\nMean metrics:")
    print(pd.DataFrame([mean_row])[metric_cols].to_string(index=False))
    print(f"Saved to: {save_path}\n")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        run_one(args, dataset_name, device)
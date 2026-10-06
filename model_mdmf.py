import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scalegcn import BiScaleGCN


def get_edge_index(matrix: torch.Tensor) -> torch.Tensor:
    """
    将稠密邻接矩阵转为 edge_index。
    """
    edge_index = torch.nonzero(matrix > 0, as_tuple=False).t().contiguous()
    return edge_index.long().to(matrix.device)


class MLP(nn.Module):
    def __init__(self, embedding_size: int, drop_rate: float):
        super().__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        h1 = max(1, self.embedding_size // 2)
        h2 = max(1, self.embedding_size // 4)
        h3 = max(1, self.embedding_size // 6)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, h1),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(h3, 1, bias=False),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features_embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp_prediction(features_embedding)


class MDMFEncoder(nn.Module):
    """
    MDMF-like 低秩特征提取分支。

    这里不破坏原 SGFCCDA 主流程，而是在 CNN 得到最终节点表示之后，
    额外加入一个由训练关联矩阵 A_train 导出的低秩表示分支：

    disease profile: S_d @ A_train       -> [n_disease, n_microbe]
    microbe profile: S_m @ A_train.T     -> [n_microbe, n_disease]

    这样做的好处是：
    1. MDMF 分支只使用当前 fold 的 train_assoc，不会使用 test edges；
    2. CVS2/CVS3 中被遮住的 disease/microbe 仍可通过相似性矩阵 S_d/S_m 获得平滑后的 profile；
    3. 最后用一个 gate 残差融合进 CNN embedding，属于“微调式加入模块”。
    """

    def __init__(
        self,
        disease_profile_size: int,
        microbe_profile_size: int,
        latent_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.disease_low_rank = nn.Sequential(
            nn.Linear(disease_profile_size, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.microbe_low_rank = nn.Sequential(
            nn.Linear(microbe_profile_size, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.disease_project = nn.Linear(latent_dim, out_dim)
        self.microbe_project = nn.Linear(latent_dim, out_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        train_assoc: torch.Tensor,
        disease_feature: torch.Tensor,
        microbe_feature: torch.Tensor,
    ):
        # train_assoc: [n_disease, n_microbe]
        # disease_feature: [n_disease, n_disease]
        # microbe_feature: [n_microbe, n_microbe]
        disease_profile = disease_feature.mm(train_assoc)
        microbe_profile = microbe_feature.mm(train_assoc.t())

        disease_latent = self.disease_low_rank(disease_profile)
        microbe_latent = self.microbe_low_rank(microbe_profile)

        disease_out = self.disease_project(disease_latent)
        microbe_out = self.microbe_project(microbe_latent)

        return disease_out, microbe_out, disease_latent, microbe_latent

    def predict_pairs_from_latent(
        self,
        disease_latent: torch.Tensor,
        microbe_latent: torch.Tensor,
        pair_index: torch.Tensor,
        batch_size: int = 8192,
    ) -> torch.Tensor:
        outputs = []
        num_pairs = pair_index.size(0)
        scale = math.sqrt(max(1, disease_latent.size(1)))

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            batch_pair = pair_index[start:end]

            disease_idx = batch_pair[:, 0]
            microbe_idx = batch_pair[:, 1]

            score = (
                disease_latent[disease_idx] * microbe_latent[microbe_idx]
            ).sum(dim=1, keepdim=True) / scale
            outputs.append(torch.sigmoid(score))

        return torch.cat(outputs, dim=0)


class Model(nn.Module):
    """
    microbe-disease 版本：
    - 原主干：similarity -> W -> BiScaleGCN -> CNN -> Hadamard -> MLP
    - 新增分支：MDMFEncoder -> gated residual fusion -> Hadamard -> MLP
    """

    def __init__(
        self,
        disease_feat_in: int,
        microbe_feat_in: int,
        feature_out: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        drop_rate: float,
        use_mdmf: bool = True,
        mdmf_dim: int = 128,
        mdmf_gate_init: float = -2.0,
    ):
        super().__init__()

        self.disease_feat_in = disease_feat_in
        self.microbe_feat_in = microbe_feat_in
        self.feature_out = feature_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.use_mdmf = use_mdmf
        self.mdmf_dim = mdmf_dim

        self.embedding = BiScaleGCN(
            in_channels=self.feature_out,
            out_channels=self.feature_out,
            hidden_channels=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.W_disease = nn.Parameter(
            torch.zeros(size=(self.disease_feat_in, self.feature_out))
        )
        self.W_microbe = nn.Parameter(
            torch.zeros(size=(self.microbe_feat_in, self.feature_out))
        )

        nn.init.xavier_uniform_(self.W_disease.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_microbe.data, gain=1.414)

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(self.num_layers + 1, 1),
                padding=0,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self._init_cnn_weights()

        # BiScaleGCN 输出形状是: [N, num_layers+1, hidden_size]
        # 经过 Conv2d(kernel=(num_layers+1,1), out_channels=6) 后，
        # 每个节点变成 6 * hidden_size 维
        self.embedding_size = 6 * self.hidden_size

        if self.use_mdmf:
            self.mdmf_encoder = MDMFEncoder(
                disease_profile_size=self.microbe_feat_in,
                microbe_profile_size=self.disease_feat_in,
                latent_dim=self.mdmf_dim,
                out_dim=self.embedding_size,
                dropout=self.drop_rate,
            )
            # gate 用 sigmoid 控制 MDMF 分支进入主干的强度。
            # 默认 sigmoid(-2) 约为 0.12，避免一开始破坏原模型。
            self.mdmf_gate = nn.Parameter(torch.tensor(float(mdmf_gate_init)))
        else:
            self.mdmf_encoder = None
            self.register_parameter("mdmf_gate", None)

        self.mlp_prediction = MLP(self.embedding_size, self.drop_rate)

    def _init_cnn_weights(self):
        for m in self.cnn_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(
        self,
        heter_adj: torch.Tensor,
        disease_feature: torch.Tensor,
        microbe_feature: torch.Tensor,
        train_assoc: torch.Tensor = None,
    ):
        """
        输出 disease embeddings 和 microbe embeddings。
        若 use_mdmf=True，需要传入当前 fold 的 train_assoc。
        """
        disease_f = disease_feature.mm(self.W_disease)
        microbe_f = microbe_feature.mm(self.W_microbe)

        n_disease = disease_f.size(0)
        n_microbe = microbe_f.size(0)
        num_nodes = n_disease + n_microbe

        node_feature = torch.cat((disease_f, microbe_f), dim=0)
        edge_index = get_edge_index(heter_adj)

        # x shape: [num_nodes, num_layers+1, hidden_size]
        x = self.embedding(node_feature, edge_index, use_softmax=False)

        # Conv2d 需要 [N, C, H, W]
        x = x.unsqueeze(1)   # [num_nodes, 1, num_layers+1, hidden_size]
        cnn_outputs = self.cnn_layer(x).view(num_nodes, -1)

        disease_embeddings = cnn_outputs[:n_disease, :]
        microbe_embeddings = cnn_outputs[n_disease:, :]

        # 在截图中 CNN 后、Hadamard 前的位置加入 MDMF 分支。
        if self.use_mdmf:
            if train_assoc is None:
                raise ValueError("use_mdmf=True 时，encode() 必须传入 train_assoc。")

            disease_mdmf, microbe_mdmf, _, _ = self.mdmf_encoder(
                train_assoc=train_assoc,
                disease_feature=disease_feature,
                microbe_feature=microbe_feature,
            )
            gate = torch.sigmoid(self.mdmf_gate)
            disease_embeddings = disease_embeddings + gate * disease_mdmf
            microbe_embeddings = microbe_embeddings + gate * microbe_mdmf

        return disease_embeddings, microbe_embeddings

    def mdmf_predict_pairs(
        self,
        train_assoc: torch.Tensor,
        disease_feature: torch.Tensor,
        microbe_feature: torch.Tensor,
        pair_index: torch.Tensor,
        batch_size: int = 8192,
    ) -> torch.Tensor:
        """
        MDMF 辅助重构分支，用于在训练时加入一个小权重的 BCE loss。
        """
        if not self.use_mdmf:
            raise ValueError("use_mdmf=False 时不能调用 mdmf_predict_pairs。")

        _, _, disease_latent, microbe_latent = self.mdmf_encoder(
            train_assoc=train_assoc,
            disease_feature=disease_feature,
            microbe_feature=microbe_feature,
        )
        return self.mdmf_encoder.predict_pairs_from_latent(
            disease_latent=disease_latent,
            microbe_latent=microbe_latent,
            pair_index=pair_index,
            batch_size=batch_size,
        )

    def predict_pairs(
        self,
        disease_embeddings: torch.Tensor,
        microbe_embeddings: torch.Tensor,
        pair_index: torch.Tensor,
        batch_size: int = 8192,
    ) -> torch.Tensor:
        """
        pair_index: [num_pairs, 2]
        每一行是 [disease_idx, microbe_idx]
        """
        outputs = []
        num_pairs = pair_index.size(0)

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            batch_pair = pair_index[start:end]

            disease_idx = batch_pair[:, 0]
            microbe_idx = batch_pair[:, 1]

            pair_feature = (
                disease_embeddings[disease_idx] * microbe_embeddings[microbe_idx]
            )
            batch_score = self.mlp_prediction(pair_feature)
            outputs.append(batch_score)

        return torch.cat(outputs, dim=0)

    def predict_matrix(
        self,
        disease_embeddings: torch.Tensor,
        microbe_embeddings: torch.Tensor,
        row_batch_size: int = 64,
    ) -> torch.Tensor:
        """
        输出完整 disease × microbe 预测矩阵
        """
        score_blocks = []
        n_disease = disease_embeddings.size(0)
        n_microbe = microbe_embeddings.size(0)
        emb_dim = disease_embeddings.size(1)

        for start in range(0, n_disease, row_batch_size):
            end = min(start + row_batch_size, n_disease)

            block_feature = (
                disease_embeddings[start:end].unsqueeze(1)
                * microbe_embeddings.unsqueeze(0)
            )
            block_feature = block_feature.reshape(-1, emb_dim)

            block_score = self.mlp_prediction(block_feature).view(end - start, n_microbe)
            score_blocks.append(block_score)

        return torch.cat(score_blocks, dim=0)

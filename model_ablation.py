import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scalegcn import BiScaleGCN
from torch_geometric.nn import GCNConv


def get_edge_index(matrix: torch.Tensor) -> torch.Tensor:
    """将稠密邻接矩阵转为 edge_index。"""
    edge_index = torch.nonzero(matrix > 0, as_tuple=False).t().contiguous()
    return edge_index.long().to(matrix.device)


class MLP(nn.Module):
    def __init__(self, embedding_size: int, drop_rate: float):
        super().__init__()
        h1 = max(1, embedding_size // 2)
        h2 = max(1, embedding_size // 4)
        h3 = max(1, embedding_size // 6)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(embedding_size, h1),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_prediction(x)


class PlainGCNBackbone(nn.Module):
    """
    普通 GCN 消融主干。

    为保证和 BiScaleGCN 后续 CNN 接口一致，返回：
        [N, num_layers + 1, hidden_size]
    即：初始映射后的表示 + 每层 GCN 的输出。
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.init_layer = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
            for _ in range(num_layers)
        ])

        nn.init.xavier_uniform_(self.init_layer.weight)
        if self.init_layer.bias is not None:
            nn.init.constant_(self.init_layer.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, use_softmax: bool = False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.init_layer(x))

        layer_outputs = [x.unsqueeze(1)]
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.leaky_relu(x, negative_slope=0.1)
            layer_outputs.append(x.unsqueeze(1))

        out = torch.cat(layer_outputs, dim=1)
        if use_softmax:
            return F.log_softmax(out, dim=1)
        return out


class MDMFEncoder(nn.Module):
    """与最终模型一致的 MDMF 低秩分支。"""
    def __init__(
        self,
        disease_profile_size: int,
        microbe_profile_size: int,
        latent_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
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

    def forward(self, train_assoc, disease_feature, microbe_feature):
        disease_profile = disease_feature.mm(train_assoc)
        microbe_profile = microbe_feature.mm(train_assoc.t())

        disease_latent = self.disease_low_rank(disease_profile)
        microbe_latent = self.microbe_low_rank(microbe_profile)
        disease_out = self.disease_project(disease_latent)
        microbe_out = self.microbe_project(microbe_latent)
        return disease_out, microbe_out, disease_latent, microbe_latent

    def predict_pairs_from_latent(self, disease_latent, microbe_latent, pair_index, batch_size=8192):
        outputs = []
        scale = math.sqrt(max(1, disease_latent.size(1)))
        for start in range(0, pair_index.size(0), batch_size):
            batch_pair = pair_index[start:start + batch_size]
            d_idx = batch_pair[:, 0]
            m_idx = batch_pair[:, 1]
            score = (disease_latent[d_idx] * microbe_latent[m_idx]).sum(dim=1, keepdim=True) / scale
            outputs.append(torch.sigmoid(score))
        return torch.cat(outputs, dim=0)


class Model(nn.Module):
    """
    消融实验模型。

    ablation:
      - full:     ScaleGCN + CNN + MDMF（完整模型）
      - no_mdmf:  ScaleGCN + CNN，关闭 MDMF
      - no_cnn:   ScaleGCN + MDMF，去掉 CNN；直接 flatten 各层 ScaleGCN 表示
      - gcn:      普通 GCN + CNN + MDMF，用普通 GCN 替换 ScaleGCN

    每个变体只改变一个核心组件，其余训练与超参数保持一致。
    """
    VALID_ABLATIONS = {"full", "no_mdmf", "no_cnn", "gcn"}

    def __init__(
        self,
        disease_feat_in: int,
        microbe_feat_in: int,
        feature_out: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        drop_rate: float,
        ablation: str = "full",
        mdmf_dim: int = 128,
        mdmf_gate_init: float = -2.0,
    ):
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation}; choose from {sorted(self.VALID_ABLATIONS)}")

        self.disease_feat_in = disease_feat_in
        self.microbe_feat_in = microbe_feat_in
        self.feature_out = feature_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.ablation = ablation

        self.use_scalegcn = ablation != "gcn"
        self.use_cnn = ablation != "no_cnn"
        self.use_mdmf = ablation != "no_mdmf"
        self.mdmf_dim = mdmf_dim

        if self.use_scalegcn:
            self.embedding = BiScaleGCN(
                in_channels=feature_out,
                out_channels=feature_out,
                hidden_channels=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.embedding = PlainGCNBackbone(
                in_channels=feature_out,
                hidden_channels=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.W_disease = nn.Parameter(torch.zeros(disease_feat_in, feature_out))
        self.W_microbe = nn.Parameter(torch.zeros(microbe_feat_in, feature_out))
        nn.init.xavier_uniform_(self.W_disease.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_microbe.data, gain=1.414)

        if self.use_cnn:
            self.cnn_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=6,
                    kernel_size=(num_layers + 1, 1),
                    padding=0,
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
            for m in self.cnn_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.embedding_size = 6 * hidden_size
        else:
            # 不使用 feature convolution：保留初始层 + 每层 ScaleGCN 表示，直接拼接后预测。
            self.cnn_layer = None
            self.embedding_size = (num_layers + 1) * hidden_size

        if self.use_mdmf:
            self.mdmf_encoder = MDMFEncoder(
                disease_profile_size=microbe_feat_in,
                microbe_profile_size=disease_feat_in,
                latent_dim=mdmf_dim,
                out_dim=self.embedding_size,
                dropout=drop_rate,
            )
            self.mdmf_gate = nn.Parameter(torch.tensor(float(mdmf_gate_init)))
        else:
            self.mdmf_encoder = None
            self.register_parameter("mdmf_gate", None)

        self.mlp_prediction = MLP(self.embedding_size, drop_rate)

    def encode(self, heter_adj, disease_feature, microbe_feature, train_assoc=None):
        disease_f = disease_feature.mm(self.W_disease)
        microbe_f = microbe_feature.mm(self.W_microbe)
        n_disease = disease_f.size(0)
        n_microbe = microbe_f.size(0)
        num_nodes = n_disease + n_microbe

        node_feature = torch.cat((disease_f, microbe_f), dim=0)
        edge_index = get_edge_index(heter_adj)

        # [N, num_layers+1, hidden_size]
        x = self.embedding(node_feature, edge_index, use_softmax=False)

        if self.use_cnn:
            # 与最终模型完全一致
            node_embeddings = self.cnn_layer(x.unsqueeze(1)).view(num_nodes, -1)
        else:
            # CNN 消融：不进行 feature convolution，直接拼接各层表示
            node_embeddings = x.reshape(num_nodes, -1)

        disease_embeddings = node_embeddings[:n_disease]
        microbe_embeddings = node_embeddings[n_disease:]

        if self.use_mdmf:
            if train_assoc is None:
                raise ValueError("启用 MDMF 时 encode() 必须传入 train_assoc")
            disease_mdmf, microbe_mdmf, _, _ = self.mdmf_encoder(
                train_assoc=train_assoc,
                disease_feature=disease_feature,
                microbe_feature=microbe_feature,
            )
            gate = torch.sigmoid(self.mdmf_gate)
            disease_embeddings = disease_embeddings + gate * disease_mdmf
            microbe_embeddings = microbe_embeddings + gate * microbe_mdmf

        return disease_embeddings, microbe_embeddings

    def mdmf_predict_pairs(self, train_assoc, disease_feature, microbe_feature, pair_index, batch_size=8192):
        if not self.use_mdmf:
            raise ValueError("no_mdmf 消融下不能调用 mdmf_predict_pairs")
        _, _, disease_latent, microbe_latent = self.mdmf_encoder(
            train_assoc=train_assoc,
            disease_feature=disease_feature,
            microbe_feature=microbe_feature,
        )
        return self.mdmf_encoder.predict_pairs_from_latent(
            disease_latent, microbe_latent, pair_index, batch_size=batch_size
        )

    def predict_pairs(self, disease_embeddings, microbe_embeddings, pair_index, batch_size=8192):
        outputs = []
        for start in range(0, pair_index.size(0), batch_size):
            batch_pair = pair_index[start:start + batch_size]
            d_idx = batch_pair[:, 0]
            m_idx = batch_pair[:, 1]
            pair_feature = disease_embeddings[d_idx] * microbe_embeddings[m_idx]
            outputs.append(self.mlp_prediction(pair_feature))
        return torch.cat(outputs, dim=0)

    def predict_matrix(self, disease_embeddings, microbe_embeddings, row_batch_size=64):
        blocks = []
        n_disease = disease_embeddings.size(0)
        n_microbe = microbe_embeddings.size(0)
        emb_dim = disease_embeddings.size(1)

        for start in range(0, n_disease, row_batch_size):
            end = min(start + row_batch_size, n_disease)
            block_feature = (
                disease_embeddings[start:end].unsqueeze(1)
                * microbe_embeddings.unsqueeze(0)
            ).reshape(-1, emb_dim)
            block_score = self.mlp_prediction(block_feature).view(end - start, n_microbe)
            blocks.append(block_score)
        return torch.cat(blocks, dim=0)

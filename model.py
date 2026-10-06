import torch
import torch.nn as nn
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


class Model(nn.Module):
    """
    改成 microbe-disease 版本：
    - 输入顺序：disease 在前，microbe 在后
    - 相似度：由外部传入 disease_gip / microbe_gip
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
    ):
        super().__init__()

        self.disease_feat_in = disease_feat_in
        self.microbe_feat_in = microbe_feat_in
        self.feature_out = feature_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.drop_rate = drop_rate

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
    ):
        """
        输出 disease embeddings 和 microbe embeddings
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

        return disease_embeddings, microbe_embeddings

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

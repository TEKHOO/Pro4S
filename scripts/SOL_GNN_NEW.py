import math
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from torch import einsum
from einops.layers.torch import Rearrange
from torch_geometric.utils import softmax, scatter
from torchmetrics.classification import Accuracy, Precision, Recall, AUROC
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import numpy as np
from torchmetrics import R2Score
import pandas as pd
import os


def augment_graph(labels, edge_index):
    N = len(labels)

    # Step 1: 调整原始边索引（后移8位）
    new_edge_index = edge_index + 8  # 新增8个节点在开头

    # Step 2: 创建局部连接
    local_edges = []
    num_classes = 7  # 现在7个类别
    for k in range(num_classes):
        mask = (labels == k)
        class_nodes = torch.where(mask)[0]
        if len(class_nodes) > 0:
            super_node_idx = k + 1  # 局部超级节点使用索引1-7
            shifted_nodes = class_nodes + 8  # 原节点后移8位

            # 创建双向连接
            src = torch.cat([
                shifted_nodes,
                torch.full_like(shifted_nodes, super_node_idx)
            ])
            dst = torch.cat([
                torch.full_like(shifted_nodes, super_node_idx),
                shifted_nodes
            ])
            local_edges.append(torch.stack([src, dst]))

    # Step 4: 创建全局连接（索引0）
    global_node_idx = 0
    all_nodes = torch.arange(N + 8).cuda()  # 总节点数 = 8新增 + N原节点

    # 双向连接全局节点
    src_global = torch.cat([all_nodes, torch.full_like(all_nodes, global_node_idx)])
    dst_global = torch.cat([torch.full_like(all_nodes, global_node_idx), all_nodes])

    # 合并所有边
    local_edge = torch.cat(local_edges, dim=1) if local_edges else torch.empty((2, 0), dtype=torch.long)
    global_edge = torch.stack([src_global, dst_global])

    return torch.cat([new_edge_index, local_edge, global_edge], dim=-1)


class GNN_no_edge(nn.Module):
    """
    无边的GNN
    """
    def __init__(self, d_emb, n_heads, dropout=0.2):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.dropout = nn.Dropout(dropout)

        # (h_j || e_ji || h_i) to scalar for each head
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, n_heads),
        )
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            Rearrange("n (h d) -> n h d", d=self.d_head),
        )
        self.to_h = nn.Linear(d_emb, d_emb, bias=False)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4), nn.ReLU(), nn.Linear(d_emb * 4, d_emb)
        )

    def forward(self, h, edge_idx):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)
        #
        # Node update
        #
        hi, hj = h[edge_idx[0]], h[edge_idx[1]]

        hi_hj = torch.cat([hi, hj], dim=-1)

        # Compute attention weights for each edge.
        w = self.att_mlp(hi_hj) / math.sqrt(self.d_head)
        att = softmax(w, dim=0, index=edge_idx[0])  # n h

        # Compute node values.
        vj = self.node_mlp(hj)  # n, n_heads, d_head

        # Aggregate node values with attention weights
        # to update node features.
        _h = einsum("nh,nhd->nhd", att, vj)
        _h = rearrange(_h, "n h d -> n (h d)")
        _h = scatter(_h, index=edge_idx[0], reduce='sum')
        _h = self.to_h(_h)  # Final linear projection.

        # FeedForward
        h = h + self.dropout(self.ff(_h))

        return h


class GNN_edge(nn.Module):  # 边更新入点
    def __init__(self, d_emb, n_heads, dropout=0.2):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.dropout = nn.Dropout(dropout)

        # (h_j || e_ji || h_i) to scalar for each head
        self.att_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, n_heads),
        )
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            Rearrange("n (h d) -> n h d", d=self.d_head),
        )
        self.to_h = nn.Linear(d_emb, d_emb, bias=False)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4), nn.ReLU(), nn.Linear(d_emb * 4, d_emb)
        )

        # (h_j || e_ji || h_i) to d_emb
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.Dropout(dropout),
        )

    def forward(self, h, e, edge_idx):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)

        #
        # Node update
        #
        hi, hj = h[edge_idx[0]], h[edge_idx[1]]

        hi_eij_hj = torch.cat([hi, e, hj], dim=-1)
        eij_hj = torch.cat([e, hj], dim=-1)

        # Compute attention weights for each edge.
        w = self.att_mlp(hi_eij_hj) / math.sqrt(self.d_head)
        att = softmax(w, dim=0, index=edge_idx[0])  # n h

        # Compute node values.
        vj = self.node_mlp(eij_hj)  # n, n_heads, d_head

        # Aggregate node values with attention weights
        # to update node features.
        _h = einsum("nh,nhd->nhd", att, vj)
        _h = rearrange(_h, "n h d -> n (h d)")
        _h = scatter(_h, index=edge_idx[0], reduce='sum')
        _h = self.to_h(_h)  # Final linear projection.

        # FeedForward
        h = h + self.dropout(self.ff(_h))

        #
        # Edge update
        #
        hi, hj = h[edge_idx[0]], h[edge_idx[1]]
        hi_eij_hj = torch.cat([hi, e, hj], dim=-1)
        e = e + self.edge_mlp(hi_eij_hj)

        return h, e

# class GNN_edge(nn.Module):  # 边不更新入点
#     def __init__(self, d_emb, n_heads, dropout=0.2):
#         super().__init__()
#
#         self.n_heads = n_heads
#         self.d_head = d_emb // n_heads
#         self.dropout = nn.Dropout(dropout)
#
#         # (h_j || e_ji || h_i) to scalar for each head
#         self.att_mlp = nn.Sequential(
#             nn.Linear(2 * d_emb, d_emb),
#             nn.ReLU(),
#             nn.Linear(d_emb, d_emb),
#             nn.ReLU(),
#             nn.Linear(d_emb, n_heads),
#         )
#         # (e_ji || h_j) to d_emb
#         self.node_mlp = nn.Sequential(
#             nn.Linear(d_emb, d_emb),
#             nn.GELU(),
#             nn.Linear(d_emb, d_emb),
#             nn.GELU(),
#             nn.Linear(d_emb, d_emb),
#             Rearrange("n (h d) -> n h d", d=self.d_head),
#         )
#         self.to_h = nn.Linear(d_emb, d_emb, bias=False)
#
#         # Feedforward
#         self.ff = nn.Sequential(
#             nn.Linear(d_emb, d_emb * 4), nn.ReLU(), nn.Linear(d_emb * 4, d_emb)
#         )
#
#         # (h_j || e_ji || h_i) to d_emb
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * d_emb, d_emb),
#             nn.GELU(),
#             nn.Linear(d_emb, d_emb),
#             nn.GELU(),
#             nn.Linear(d_emb, d_emb),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, h, e, edge_idx):
#         # h: (# nodes, d_emb)
#         # e: (# edges, d_emb)
#         # edge_index: (2, # edges)
#
#         #
#         # Node update
#         #
#         hi, hj = h[edge_idx[0]], h[edge_idx[1]]
#
#         hi_eij_hj = torch.cat([hi, hj], dim=-1)
#
#         # Compute attention weights for each edge.
#         w = self.att_mlp(hi_eij_hj) / math.sqrt(self.d_head)
#         att = softmax(w, dim=0, index=edge_idx[0])  # n h
#
#         # Compute node values.
#         vj = self.node_mlp(hj)  # n, n_heads, d_head
#
#         # Aggregate node values with attention weights
#         # to update node features.
#         _h = einsum("nh,nhd->nhd", att, vj)
#         _h = rearrange(_h, "n h d -> n (h d)")
#         _h = scatter(_h, index=edge_idx[0], reduce='sum')
#         _h = self.to_h(_h)  # Final linear projection.
#
#         # FeedForward
#         h = h + self.dropout(self.ff(_h))
#
#         #
#         # Edge update
#         #
#         hi, hj = h[edge_idx[0]], h[edge_idx[1]]
#         hi_eij_hj = torch.cat([hi, hj], dim=-1)
#         e = e + self.dropout(self.edge_mlp(hi_eij_hj))
#
#         return h, e


class GNN_inter_edge(nn.Module):
    def __init__(self, d_emb, n_heads, dropout=0.2):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.dropout = nn.Dropout(dropout)

        #
        # structure
        #
        # (h_j || e_ji || h_i) to scalar for each head
        self.att_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, n_heads),
        )
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            Rearrange("n (h d) -> n h d", d=self.d_head),
        )
        self.to_h = nn.Linear(d_emb, d_emb, bias=False)
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4), nn.ReLU(), nn.Linear(d_emb * 4, d_emb)
        )

        #
        # surface
        #
        # (e_ji || h_j) to d_emb
        self.node_mlp_sf = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
        )
        self.to_h_sf = nn.Linear(d_emb, d_emb, bias=False)
        # Feedforward
        self.ff_sf = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4), nn.ReLU(), nn.Linear(d_emb * 4, d_emb)
        )

        #
        # edge
        #
        # (h_j || e_ji || h_i) to d_emb
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.Dropout(dropout),
        )

    def forward(self, h_st, e, h_sf, edge_idx):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)
        """
        edge_index: eg:[[   0    1    2 ... 3067 3068 3069], [ 116    9  126 ...  123  108  118]]
        edge_index[0] = surface
        edge_index[1] = structure
        """
        #
        # Node update
        #
        hsf, hst = h_sf[edge_idx[0]], h_st[edge_idx[1]]
        hi_eij_hj = torch.cat([hsf, e, hst], dim=-1)

        ### structure
        eij_hsf = torch.cat([e, hsf], dim=-1)

        # Compute attention weights for each edge.
        wst = self.att_mlp(hi_eij_hj) / math.sqrt(self.d_head)
        attst = softmax(wst, dim=0, index=edge_idx[1])  # n h

        # Compute node values.
        vst = self.node_mlp(eij_hsf)  # n, n_heads, d_head

        ### surface
        vsf = self.node_mlp_sf(hi_eij_hj)  # n, n_heads, d_head

        # Aggregate node values with attention weights
        # to update node features.(structure)
        num_nodes_st = h_st.shape[0]
        _h = einsum("nh,nhd->nhd", attst, vst)
        _h = rearrange(_h, "n h d -> n (h d)")
        _h = scatter(_h, index=edge_idx[1], dim_size=num_nodes_st, reduce='sum')
        _h = self.to_h(_h)  # Final linear projection.
        # FeedForward
        h_st = h_st + self.dropout(self.ff(_h))

        # to update node features.(surface)
        _h = self.to_h_sf(vsf)  # Final linear projection.
        # FeedForward
        h_sf = h_sf + self.dropout(self.ff_sf(_h))

        #
        # Edge update
        #
        hi, hj = h_sf[edge_idx[0]], h_st[edge_idx[1]]
        hi_eij_hj = torch.cat([hi, e, hj], dim=-1)
        e = e + self.edge_mlp(hi_eij_hj)

        return h_st, e, h_sf


class THREE_D_ENCODER(nn.Module):
    def __init__(self, d_emb, n_heads, dropout=0.5):
        super().__init__()
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.dropout = dropout
        self.residue_encoder = GNN_no_edge(d_emb, n_heads, dropout)
        self.surface_encoder = GNN_edge(d_emb, n_heads, dropout)
        self.inter_module = GNN_inter_edge(d_emb, n_heads, dropout)

    def forward(self, h_st, h_sf, e_st, e_sf, e_inter, edge_idx_inter, edge_idx_structure, edge_idx_surface):
        h_st = self.residue_encoder(h_st, edge_idx_structure)
        h_sf, e_sf = self.surface_encoder(h_sf, e_sf, edge_idx_surface)
        h_st, e_inter, h_sf = self.inter_module(h_st, e_inter, h_sf, edge_idx_inter)

        return h_st, h_sf, e_st, e_sf, e_inter, edge_idx_inter, edge_idx_structure, edge_idx_surface


class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim, num_latent_queries, num_heads):
        super().__init__()
        self.num_latent_queries = num_latent_queries
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True  # 启用 batch_first
        )

    def forward(self, x):
        B, N, C = x.shape
        # 扩展 latent_queries 到 (B, num_latent_queries, C)
        mean_info = torch.mean(x, dim=1, keepdim=True)
        attn_output, _ = self.multihead_attn(mean_info, x, x)
        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_emb, n_heads, dropout=0.5):
        super().__init__()
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_emb,
            num_heads=n_heads,
            batch_first=True,
        )
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),  # 扩展维度
            nn.ReLU(),
            nn.Linear(4 * d_emb, d_emb),  # 恢复维度
        )
        # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接 1
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)  # 残差连接 + Dropout

        # 残差连接 2
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)   # 残差连接 + Dropout
        return x


# class LocalSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_local_cls):
#         super(LocalSelfAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_local_cls = num_local_cls
#         # 初始化局部 cls token 参数，形状为 (num_local_cls, embed_dim)
#         self.local_cls = nn.Parameter(torch.randn(num_local_cls, embed_dim))
#         # 使用 batch_first=True，使得输入输出形状为 (B, L, D)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#
#     def forward(self, seq, local_labels):
#         """
#         seq: Tensor, shape (B, L, D)
#         local_labels: Tensor, shape (L,) —— 假设每个 batch 的局部标签相同
#         返回每个局部区域更新后的 cls token 表示，形状为 (B, num_groups, D)
#         """
#         B, L, D = seq.shape
#         local_features = []
#
#         # 遍历所有局部 cls token（编号从 0 到 num_local_cls-1）
#         for cls_idx in range(self.num_local_cls):
#             # 找到该局部区域对应的 token 索引（局部标签在所有样本中相同）
#             idx = (local_labels == cls_idx).nonzero(as_tuple=True)[0]
#             if idx.numel() == 0:
#                 continue  # 若该区域没有 token，则跳过
#
#             # 从 seq 中选取该组 token，利用 batch 索引: 形状 (B, n_group, D)
#             group_tokens = seq[:, idx, :]
#             # 获取对应的局部 cls token，并扩展到 batch 维度：形状 (B, 1, D)
#             cls_token = self.local_cls[cls_idx].unsqueeze(0).unsqueeze(1).expand(B, 1, D)
#             # 拼接局部 cls token 与组内 token，形成新的序列：形状 (B, n_group+1, D)
#             group_input = torch.cat([cls_token, group_tokens], dim=1)
#             # 对该局部组进行 self-attention
#             group_out, _ = self.attn(group_input, group_input, group_input)
#             # 更新后的 cls token 为每个 batch 的第一个 token：形状 (B, D)
#             local_features.append(group_out[:, 0, :])
#
#         # 堆叠所有局部区域的 cls 表示，结果形状为 (B, num_groups, D)
#         local_features = torch.stack(local_features, dim=1)
#         return local_features


class Fusion_Model(nn.Module):
    def __init__(self, d_emb, n_heads, num_structure_layer, num_seq_layer, num_fusion_layer, d_seq, d_structure_n, d_surface_n, d_structure_e=504, dropout=0.5, queue_size=256, **kwargs):
        """ Graph labeling network """
        super(Fusion_Model, self).__init__()
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.num_structure_layer = num_structure_layer
        self.num_fusion_layer = num_fusion_layer
        self.num_seq_layer = num_seq_layer
        self.rbf = 16

        self.decoder_projection = nn.Sequential(
            nn.Linear(d_seq + 256, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb)
        )
        # self.seq_proj = nn.Linear(d_seq, d_emb, bias=False)

        self.structure_n_projection = nn.Sequential(
            nn.Linear(d_structure_n, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )

        self.surface_n_projection = nn.Sequential(
            nn.Linear(d_surface_n, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )

        self.structure_e_projection = nn.Sequential(
            nn.Linear(d_structure_e, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb)
        )

        self.distance_projection = nn.Sequential(
            nn.Linear(self.rbf, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
        )

        self.lower_projection = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, 256),
        )

        self.st_layers = nn.ModuleList(
            [
                THREE_D_ENCODER(d_emb=d_emb, n_heads=n_heads, dropout=dropout)
                for _ in range(self.num_structure_layer)
            ]
        )
        
        # self.seq_layers = nn.ModuleList(
        #     [
        #         TransformerEncoderLayer(d_emb, n_heads)
        #         for _ in range(self.num_seq_layer)
        #     ]
        # )
        #
        # self.attention_pool = AttentionPoolingLayer(input_dim=d_emb, num_latent_queries=1, num_heads=n_heads)
        #
        # decoder_layer = TransformerDecoderLayer(d_model=d_emb, nhead=n_heads, dropout=dropout, batch_first=True)
        # self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_fusion_layer)

        self.decoder = nn.ModuleList(
            [
                GNN_edge(d_emb=d_emb, n_heads=n_heads, dropout=dropout)
                for _ in range(self.num_fusion_layer)
            ]
        )
        self.out_gate_proj = nn.Sequential(
            nn.Linear(1, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Linear(d_emb, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.mlp_struct = nn.Linear(d_emb, d_emb)
        self.mlp_seq = nn.Linear(d_seq, d_emb)
        # self.structure_cls_token = nn.Parameter(torch.zeros(1 + 7, self.d_emb))
        # self.seq_cls_token = nn.Parameter(torch.zeros(1, self.d_emb))

        # Temperature for contrastive learning
        self.temperature = 0.1
        self.queue_size = queue_size

        # Initialize queues for structural and seq CLS tokens
        self.struct_queue = nn.Parameter(torch.zeros(queue_size, d_emb), requires_grad=False)
        self.seq_queue = nn.Parameter(torch.zeros(queue_size, d_emb), requires_grad=False)
        self.queue_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

        self._init_params()

    def forward(self, batch):
        # node
        seq = batch['seq_feature']
        structure_node = batch['structure_node_feature']
        surface_node = batch['surface_node_feature']
        # seq = self.seq_projection(seq.squeeze(0))
        seq = seq.squeeze(0)
        n_st = self.structure_n_projection(structure_node.squeeze(0))
        n_sf = self.surface_n_projection(surface_node.squeeze(0))
        sasa = batch['sasa'].squeeze(0)
        plddt = batch['plddt'].squeeze(0)

        # edge
        structure_edge = batch['structure_edge_feature']
        surface_edge = batch['surface_edge_feature']
        inter_edge = batch['inter_edge_feature']
        e_st = self.structure_e_projection(structure_edge.squeeze(0))
        e_sf = self.distance_projection(surface_edge.squeeze(0))
        e_inter = self.distance_projection(inter_edge.squeeze(0))

        # index
        edge_idx_structure = batch['edge_idx_structure'].squeeze(0)  # 原始的edge_index
        edge_idx_structure_with_global = batch['edge_idx_structure_with_global'].squeeze(0)
        edge_idx_surface = batch['edge_idx_surface'].squeeze(0)
        # aa_local_label = batch['aa_local_label'].squeeze(0)
        edge_idx_inter = batch['edge_idx_inter'].squeeze(0)
        edge_idx_inter[1] += 1  # 考虑全局节点的edge_idx
        # edge_idx_structure = augment_graph(aa_local_label, _edge_idx_structure)  # 含有全局连接和局部连接的edge_index

        #
        #  structure encoder
        #
        structure_cls_token = torch.mean(n_st, dim=0, keepdim=True)
        n_st = torch.cat([structure_cls_token, n_st], dim=0)
        for st_layer in self.st_layers:
            n_st, n_sf, e_st, e_sf, e_inter, edge_idx_inter, edge_idx_structure_with_global, edge_idx_surface = st_layer(n_st, n_sf, e_st, e_sf, e_inter, edge_idx_inter, edge_idx_structure_with_global, edge_idx_surface)   # h_st, h_sf, e_st, e_sf, e_inter, edge_idx_inter, edge_idx_structure, edge_idx_surface

        struct_cls_tokens = n_st.unsqueeze(0)[:, 0, :]  # 1, d_emb
        n_st = n_st[1:, :]

        #
        #  seq encoder
        #
        # seq_cls_token = torch.mean(seq, dim=0, keepdim=True)
        # _seq = torch.cat([seq_cls_token, seq], dim=0)
        # _seq = _seq.unsqueeze(0)
        # for seq_layer in self.seq_layers:
        #     seq = seq_layer(seq)
        # # global emb
        # global_feature_seq = _seq[:, 0, :]
        # seq = _seq[:, 1:, :]
        # # local emb
        # seq = seq.squeeze(0)
        # d_emb = seq.size(1)
        # local_feature = []
        # for label in range(7):  # 遍历0到6的标签，确保顺序
        #     # 获取当前标签对应的嵌入向量
        #     mask = (aa_local_label == label)
        #     embeddings = seq[mask]
        #
        #     # 计算均值（若无数据则用零填充）
        #     if embeddings.size(0) == 0:
        #         mean_embed = torch.zeros(1, d_emb, device=seq.device)
        #     else:
        #         mean_embed = self.attention_pool(embeddings.unsqueeze(0)).squeeze(0)
        #
        #     local_feature.append(mean_embed)
        #
        # # 拼接所有local_tokens
        # local_feature_seq = torch.cat(local_feature, dim=0).unsqueeze(0)
        # # 拼接所有tokens
        # seq = torch.cat([_seq[:, :1, :], local_feature_seq, _seq[:, 1:, :]], dim=1)

        #
        # Fusion_layer
        #
        # decoder_output = self.transformer_decoder(
        #     seq, n_st,
        # )

        global_feature_seq = torch.mean(seq, dim=0, keepdim=True)  # 1, d_seq

        #  Fusion_layer
        n_st = self.lower_projection(n_st)
        node_in = torch.cat([seq, n_st], dim=-1)
        h = self.decoder_projection(node_in)
        e = e_st

        for layer in self.decoder:
            h, e = layer(h, e, edge_idx_structure)

        out_gate = torch.cat([sasa], dim=-1)
        out_gate = self.out_gate_proj(out_gate)
        decoder_output = out_gate * h
        node_mean = torch.mean(decoder_output, dim=0)

        # Predict labels using MLP
        logits = self.mlp(node_mean)

        # # Contrastive learning
        # MLP for Contrastive learning
        struct_cls_tokens = self.mlp_struct(struct_cls_tokens)
        global_feature_seq = self.mlp_seq(global_feature_seq)
        contrastive_loss_global = self._contrastive_loss(struct_cls_tokens,
                                                         global_feature_seq)  # Global CLS
        # contrastive_loss_subarea = self._contrastive_loss_subarea(struct_cls_tokens[:, 1:, :],
        #                                                           local_feature_seq)  # Subarea CLS
        #
        # Update queues with current batch global CLS tokens
        self._dequeue_and_enqueue(struct_cls_tokens, global_feature_seq)

        return {'out': logits, 'contrastive_loss_global': contrastive_loss_global}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, struct_cls_token, seq_cls_token):
        """Append new CLS tokens to the queue and dequeue older ones."""
        batch_size = struct_cls_token.size(0)

        # Get current position in the queue
        ptr = int(self.queue_ptr)

        # Replace oldest entries with the new ones
        if ptr + batch_size > self.queue_size:
            ptr = 0
        self.struct_queue[ptr:ptr + batch_size, :] = struct_cls_token
        self.seq_queue[ptr:ptr + batch_size, :] = seq_cls_token

        # Move pointer and wrap-around if necessary
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _contrastive_loss(self, struct_cls_token, seq_cls_token):
        """Compute NT-Xent contrastive loss using queue-based negative sampling."""
        batch_size = struct_cls_token.size(0)

        # Normalize CLS tokens
        z_i = F.normalize(struct_cls_token, dim=-1)
        z_j = F.normalize(seq_cls_token, dim=-1)

        # Normalize queue embeddings
        struct_queue_norm = F.normalize(self.struct_queue.clone().detach(), dim=-1)
        seq_queue_norm = F.normalize(self.seq_queue.clone().detach(), dim=-1)

        # Cosine similarity between current CLS tokens
        sim_ij = torch.matmul(z_i, z_j.T) / self.temperature  # (batch_size, batch_size)

        # Cosine similarity with negative samples from the queue
        sim_i_struct_queue = torch.matmul(z_i, seq_queue_norm.T) / self.temperature  # (batch_size, queue_size)
        sim_j_seq_queue = torch.matmul(z_j, struct_queue_norm.T) / self.temperature  # (batch_size, queue_size)

        # Combine positive and negative samples
        sim_matrix_i = torch.cat([sim_ij, sim_i_struct_queue], dim=1)  # (batch_size, batch_size + queue_size)
        sim_matrix_j = torch.cat([sim_ij.T, sim_j_seq_queue], dim=1)  # (batch_size, batch_size + queue_size)

        # Create labels (positive samples on the diagonal)
        labels = torch.arange(batch_size).long().to(sim_matrix_i.device)

        # Contrastive loss for both modalities
        loss_i = F.cross_entropy(sim_matrix_i, labels)
        loss_j = F.cross_entropy(sim_matrix_j, labels)

        loss = (loss_i + loss_j) / 2.0
        return loss

    # def _contrastive_loss_subarea(self, struct_subarea_cls_tokens, seq_subarea_cls_tokens):
    #     """Compute contrastive loss for the subarea CLS tokens without using a queue, using only the current batch."""
    #     batch_size, num_subareas, hidden_dim = struct_subarea_cls_tokens.size()
    #
    #     # Normalize CLS tokens
    #     z_i = F.normalize(struct_subarea_cls_tokens, dim=-1)
    #     z_j = F.normalize(seq_subarea_cls_tokens, dim=-1)
    #
    #     # Cosine similarity within the batch for subarea CLS tokens
    #     sim_ij = torch.matmul(z_i, z_j.transpose(1, 2)) / self.temperature  # (batch_size, num_subareas, num_subareas)
    #
    #     # Create labels (positive samples on the diagonal)
    #     labels = torch.arange(num_subareas).long().to(sim_ij.device).unsqueeze(0).expand(batch_size, -1)
    #
    #     # Reshape sim_ij and labels for efficient cross-entropy calculation
    #     sim_ij = sim_ij.view(batch_size * num_subareas, num_subareas)  # (batch_size * num_subareas, num_subareas)
    #     labels = labels.reshape(batch_size * num_subareas)  # (batch_size * num_subareas,)
    #
    #     # Compute contrastive loss in one step
    #     loss = F.cross_entropy(sim_ij, labels)
    #
    #     return loss


    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class QJGNN(pl.LightningModule):
    def __init__(
        self,
        d_seq=2560,
        d_structure_n=36,
        d_surface_n=7,
        d_structure_e=504,
        d_emb=64,
        n_heads=4,
        drop_out=0.5,
        num_structure_layer=3,
        num_seq_layer=3,
        num_fusion_layer=3,
        queue_size=256,
        global_cls=1,
        local_cls=1,
        lr=1e-3,
    ):
        super().__init__()  # def __init__(self, d_emb, n_heads, n_structure_layer, d_node, dropout=0.5, queue_size=64, **kwargs):
        self.model = Fusion_Model(d_emb, n_heads, num_structure_layer, num_seq_layer, num_fusion_layer, d_seq, d_structure_n, d_surface_n, d_structure_e=d_structure_e, dropout=drop_out, queue_size=queue_size)
        self.global_cls = global_cls
        self.local_cls = local_cls
        self.lr = lr

        self.criterion = nn.BCELoss(reduction="mean")

        # 评估指标
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.auroc = AUROC(task="binary")

        self.acc_intest = Accuracy(task="binary")
        self.auroc_intest = AUROC(task="binary")
        self.acc_test1 = Accuracy(task="binary")
        self.auroc_test1 = AUROC(task="binary")

        self.r2_score = R2Score()

        self.acc_test3 = Accuracy(task="binary")
        self.auroc_test3 = AUROC(task="binary")
        
        self.out0 = []
        self.out1 = []
        self.out2 = []
        self.out_in = []

        # Empty list for validation recovery metrics.
        self.validation_step_outputs = []

    def forward(self, batch):
        result = self.model(batch)

        return result

    def training_step(self, batch):
        result = self.forward(batch)
        target = batch['solubility']
        target = target.squeeze(0)
        out = result['out']

        loss1 = self.criterion(out, target)
        loss_g = self.global_cls * result['contrastive_loss_global']
        # loss_local = self.local_cls * result['contrastive_loss_subarea']

        loss = loss1 + loss_g

        self.log_dict(
            {
                "train/loss": loss,
                # "train/perplexity": torch.exp(loss),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            #batch_size=self.bsz,
        )

        return loss

    def validation_step(self, batch):
        result = self.forward(batch)
        target = batch['solubility']
        target = target.squeeze(0)
        out = result['out']

        loss1 = self.criterion(out, target)
        loss_g = self.global_cls * result['contrastive_loss_global']
        # loss_local = self.local_cls * result['contrastive_loss_subarea']

        loss = loss1 + loss_g

        # 更新指标
        self.accuracy.update(torch.tensor(out), torch.tensor(target))
        self.precision.update(torch.tensor(out), torch.tensor(target))
        self.recall.update(torch.tensor(out), torch.tensor(target))
        self.auroc.update(torch.tensor(out), torch.tensor(target))

        self.log_dict(
            {
                "val/loss": loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_validation_epoch_end(self):
        # 计算评价指标
        val_accuracy = self.accuracy.compute()
        val_precision = self.precision.compute()
        val_recall = self.recall.compute()
        val_auroc = self.auroc.compute()

        print(f'val_AUROC {val_auroc}')

        # 记录日志
        self.log_dict(
            {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_auroc": val_auroc,
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=True
        )

        self.validation_step_outputs.append(val_accuracy)
        self.validation_step_outputs.append(val_auroc)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.forward(batch)
        target = batch['solubility']
        name = batch['group_key'][0]
        target = target.squeeze(0)
        out = result['out']

        self.out_in.append([name, torch.tensor(out).item(), torch.tensor(target).item()])

        # loss1 = self.criterion(out, target)
        # loss_g = self.global_cls * result['contrastive_loss_global']
        # # loss_local = self.local_cls * result['contrastive_loss_subarea']
        #
        # loss = loss1 + loss_g
        #
        # # 根据 dataloader_idx 选择对应的测试集并记录指标
        # if dataloader_idx == 0:
        #     # 对 test_loader1 的指标进行更新
        #     self.acc_intest.update(torch.tensor(out), torch.tensor(target))
        #     self.auroc_intest.update(torch.tensor(out), torch.tensor(target))
        #     self.out_in.append([name, torch.tensor(out).item(), torch.tensor(target).item()])
        #
        # elif dataloader_idx == 1:
        #     # 对 test_loader2 的指标进行更新
        #     self.acc_test1.update(torch.tensor(out), torch.tensor(target))
        #     self.auroc_test1.update(torch.tensor(out), torch.tensor(target))
        #     self.out0.append([name, torch.tensor(out).item(), torch.tensor(target).item()])
        #
        # elif dataloader_idx == 2:
        #     # 对 test_loader3 的指标进行更新
        #     self.r2_score.update(torch.tensor(out), torch.tensor(target))
        #     self.out1.append([name, torch.tensor(out).item(), torch.tensor(target).item()])
        # elif dataloader_idx == 3:
        #     # 对 test_loader4 的指标进行更新
        #     self.acc_test3.update(torch.tensor(out), torch.tensor(target))
        #     self.auroc_test3.update(torch.tensor(out), torch.tensor(target))
        #     self.out2.append([name, torch.tensor(out).item(), torch.tensor(target).item()])

        # return {"loss": loss}

    def on_test_epoch_end(self):
        df = pd.DataFrame(self.out_in, columns=['name', 'Prediction', 'Label'])
        os.makedirs('../test/result', exist_ok=True)
        df.to_excel(f'../test/result/test.xlsx', index=False)
        self.out_in.clear()

    # def on_test_epoch_end(self):
    #     acc_intest = self.acc_intest.compute()
    #     auroc_intest = self.auroc_intest.compute()
    #     self.log("internal_test/accuracy", acc_intest, prog_bar=True)
    #     self.log("internal_test/auroc", auroc_intest, prog_bar=True)
    #     self.acc_intest.reset()  # 重置 accuracy
    #     self.auroc_intest.reset()  # 重置 auroc
    #
    #     acc_test1 = self.acc_test1.compute()
    #     auroc_test1 = self.auroc_test1.compute()
    #     self.log("test1/accuracy", acc_test1, prog_bar=True)
    #     self.log("test1/auroc", auroc_test1, prog_bar=True)
    #     self.acc_test1.reset()  # 重置 accuracy
    #     self.auroc_test1.reset()  # 重置 auroc
    #
    #     r2 = self.r2_score.compute()
    #     self.log("test2/r2_score", r2, prog_bar=True)
    #     self.r2_score.reset()  # 重置 r2_score
    #
    #     acc_test3 = self.acc_test3.compute()
    #     auroc_test3 = self.auroc_test3.compute()
    #     self.log("test3/accuracy", acc_test3, prog_bar=True)
    #     self.log("test3/auroc", auroc_test3, prog_bar=True)
    #     self.acc_test3.reset()  # 重置 accuracy
    #     self.auroc_test3.reset()  # 重置 auroc
    #
    #     df = pd.DataFrame(self.out_in, columns=['名称', 'Prediction', 'Label'])
    #     df.to_excel(f'./excel_out/internal_test_AUC_{auroc_intest:.4f}.xlsx', index=False)
    #     self.out_in.clear()
    #
    #     df = pd.DataFrame(self.out0, columns=['名称', 'Prediction', 'Label'])
    #     df.to_excel(f'./excel_out/test1_AUC_{auroc_test1:.4f}.xlsx', index=False)
    #     self.out0.clear()
    #
    #     df = pd.DataFrame(self.out1, columns=['名称', 'Prediction', 'Label'])
    #     df.to_excel(f'./excel_out/test2_R2_{r2:.4f}.xlsx', index=False)
    #     self.out1.clear()
    #
    #     df = pd.DataFrame(self.out2, columns=['名称', 'Prediction', 'Label'])
    #     df.to_excel(f'./excel_out/test3_AUC_{auroc_test3:.4f}.xlsx', index=False)
    #     self.out2.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

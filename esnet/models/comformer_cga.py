"""Implementation based on the template of Matformer."""

from typing import Tuple
import math
import numpy as np
import torch
import torch.nn.functional as F
# from pydantic.typing import Literal
from typing_extensions import Literal
from torch import nn
from esnet.models.utils import RBFExpansion
from esnet.utils import BaseSettings
from esnet.features import angle_emb_mp
from torch_scatter import scatter
from esnet.models.transformer import ComformerConv, ComformerConv_edge, ComformerConvEqui
from esnet.models.fusion import MultiHeadAttention, CrossAttention, CGAFusion

class iComformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["iComformer"]
    conv_layers: int = 4
    edge_layers: int = 1
    atom_input_features: int = 70
    kge_input_features: int = 128
    # funsion_feature: int = 512
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class eComformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["eComformer"]
    conv_layers: int = 3
    edge_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
            torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


class eComformer(nn.Module):  # eComFormer
    """att pyg implementation."""

    def __init__(self, config: eComformerConfig = eComformerConfig(name="eComformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.equi_update = ComformerConvEqui(in_channels=config.node_features, out_channels=config.node_features,
                                             edge_dim=config.node_features, use_second_order_repr=True)

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, _, _ = data
        node_features = self.atom_embedding(data.x)
        n_nodes = node_features.shape[0]
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)

        return torch.squeeze(out)


'''
class iComformer(nn.Module):  # iComFormer
    """att pyg implementation."""

    def __init__(self, config: iComformerConfig = iComformerConfig(name="iComformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            nn.Linear(config.triplet_input_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(in_channels=config.node_features, out_channels=config.node_features,
                                                    heads=config.node_layer_head, edge_dim=config.node_features)

        # 引入知识图谱
        self.fusion_embedding = nn.Linear(
            config.kge_input_features, config.node_features
        )
        self.kge_fc = nn.Sequential(
            nn.Linear(config.funsion_feature, config.fc_features), nn.SiLU()
        )
        self.fusion_layer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=config.funsion_feature, nhead=16, dropout=0.1,
                                           activation=nn.SiLU(),
                                           batch_first=True, norm_first=True)
                for _ in range(4)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, ldata, kgedata, lattice = data
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)  # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1)  # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei,
                                     data.edge_attr.unsqueeze(1).repeat(1, 3, 1))  # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        graph_features = self.fc(features)  # 输出64x256

        lkgedata = kgedata.x
        features1 = scatter(lkgedata, kgedata.batch, dim=0, reduce="mean")
        kge_features = self.fusion_embedding(features1)
        combined_features = torch.cat((graph_features, kge_features), dim=1)
        for i in range(4):
            combined_features = self.fusion_layer[i](combined_features)
        combined_features = self.kge_fc(combined_features)

        # out = self.fc_out(features)
        out = self.fc_out(combined_features)
        if self.link:
            out = self.link(out)

        return torch.squeeze(out)
'''


class iComformer(nn.Module):  # iComFormer
    """att pyg implementation."""

    def __init__(self, config: iComformerConfig = iComformerConfig(name="iComformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            nn.Linear(config.triplet_input_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(in_channels=config.node_features,
                                                    out_channels=config.node_features,
                                                    heads=config.node_layer_head, edge_dim=config.node_features)

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.fc1 = nn.Linear(
            config.kge_input_features, config.node_features
        )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    # 引入元素特征
        self.element_atten = MultiHeadAttention(in_dim=config.kge_input_features,
                                                k_dim=config.node_features,
                                                v_dim=config.node_features,
                                                num_heads=1)

        self.cross_atten = CrossAttention(in_dim1=config.node_features,
                                          in_dim2=config.node_features,
                                          k_dim=config.node_features,
                                          v_dim=config.node_features,
                                          num_heads=1)
        self.project_node_feats = nn.Sequential(
            nn.Linear(config.kge_input_features, config.node_features),
            nn.Softplus()
        )
        self.cga = CGAFusion(config.node_features)
        # self.cga = CGAFusion1(config.node_features)

    def forward(self, data) -> torch.Tensor:
        data, ldata, kgedata, lattice = data
        # print("data.x: ", data.x.shape)
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)  # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1)  # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1))  # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        # crystal-level readout
        graph_features = scatter(node_features, data.batch, dim=0, reduce="mean") # 使用均值聚合节点特征

        element_data = kgedata.x
        # element_feature = self.element_atten(element_data)
        # element_features = scatter(element_data, kgedata.batch, dim=0, reduce="max")
        element_features = self.project_node_feats(element_data)
        element_features = scatter(element_features, kgedata.batch, dim=0, reduce="mean")  # 64x256
        # element_feature =  self.fc1(element_features)

        # 交叉注意力机制
        # features = self.cross_atten(element_features, graph_features)

        # CGA融合
        features = self.cga(graph_features, element_features)
        
        features = self.fc(features)

        out = self.fc_out(features)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)



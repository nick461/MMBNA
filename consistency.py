import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
import numpy as np
import os
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ViewConsistencyVAE(nn.Module):
    def __init__(self, input_shape, hidden_dim=137, latent_dim=256, num_views=9,
                 beta=1e-6, gamma=0.1, num_gin_layers=3, num_classes=2,
                 cls_weight=1.0, ablation_single_view=False, dropout=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_views = 1 if ablation_single_view else num_views  # 修改这里
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.ablation_single_view = ablation_single_view  # 添加标志位

        # 获取输入维度
        self.n_timepoints = input_shape[-1]  # 时间点数量
        self.n_rois = input_shape[-2]  # ROI数量

        # 使用单个共享的残差层
        self.shared_residual = nn.Linear(self.n_timepoints, hidden_dim)

        # GIN编码器 - 修改输入维度
        self.gin_encoders = nn.ModuleList([
            nn.ModuleList([
                GINConv(
                    nn=nn.Sequential(
                        nn.Linear(self.n_timepoints if i == 0 else self.n_timepoints, self.n_timepoints),
                        nn.BatchNorm1d(self.n_timepoints),
                        nn.LeakyReLU(),
                        nn.Linear(self.n_timepoints, self.n_timepoints)
                    ),
                    train_eps=True
                ) for i in range(num_gin_layers)
            ]) for _ in range(num_views)
        ])

        # mu和logvar层
        self.fc_mu = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim) for _ in range(num_views)
        ])

        self.fc_logvar = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim) for _ in range(num_views)
        ])

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 修改分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 修改解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, self.n_rois * self.n_timepoints),
                nn.Sigmoid()
            ) for _ in range(num_views // 3)  # 只为特征视图创建解码器
        ])

        # 添加注意力投影层
        self.query_proj = nn.Linear(self.n_timepoints, self.n_timepoints)
        self.key_proj = nn.Linear(self.n_timepoints, self.n_timepoints)
        self.value_proj = nn.Linear(self.n_timepoints, self.n_timepoints)

        # 添加ResNet编码器
        self.res_encoders = nn.ModuleList([
            nn.Sequential(
                ResBlock(self.n_timepoints, hidden_dim),
                ResBlock(hidden_dim, hidden_dim),
                ResBlock(hidden_dim, hidden_dim)
            ) for _ in range(num_views)
        ])

        # 融合不同编码器的输出
        self.fusion_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 2是因为现在只有GIN和ResNet两种编码器
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.4)
        )

        # 修改投影层的维度
        # 输入维度是 timepoints (137)，输出维度是 hidden_dim (137)
        self.gin_proj = nn.Linear(self.n_timepoints, hidden_dim)

        self.cls_weight = cls_weight  # 使用传入的分类权重

        # 增加 dropout
        self.dropout = nn.Dropout(0.4)
        # 增加 L2 正则化权重
        self.weight_decay = 1e-4

    def encode(self, x, batch_idx, node_view, edge_view):
        """
        Args:
            x: [batch_size * n_rois, timepoints] 展平的特征
            batch_idx: 当前处理的视图组索引
            node_view: [batch_size, n_rois, n_rois] 节点掩码后的相关性矩阵
            edge_view: [batch_size, n_rois, n_rois] 边掩码后的相关性矩阵
        """
        batch_size = node_view.size(0)
        n_rois = node_view.size(1)

        # 重塑输入
        x = x.reshape(batch_size, n_rois, -1)  # [batch_size, n_rois, timepoints]

        # 1. 首先通过GIN处理
        gin_out = x  # [batch_size, n_rois, timepoints]

        for i, gin in enumerate(self.gin_encoders[batch_idx]):
            # 同时应用edge_view和node_view的消息传递
            edge_message = torch.bmm(edge_view, gin_out)  # 边的消息传递
            node_message = torch.bmm(node_view, gin_out)  # 节点的消息传递
            gin_out = (edge_message + node_message) / 2  # 组合两种消息

            # 重塑以适应MLP
            gin_out = gin_out.reshape(-1, self.n_timepoints)  # [batch_size * n_rois, timepoints]
            gin_out = gin.nn[0](gin_out)
            gin_out = gin.nn[1](gin_out)
            gin_out = gin.nn[2](gin_out)
            gin_out = gin.nn[3](gin_out)
            gin_out = gin_out.reshape(batch_size, n_rois, -1)  # [batch_size, n_rois, timepoints]

            # 激活和Dropout
            gin_out = F.relu(gin_out)
            if i < len(self.gin_encoders[batch_idx]) - 1:
                gin_out = F.dropout(gin_out, p=0.5, training=self.training)

        # 2. 将GIN的输出通过ResNet处理
        gin_out = gin_out.reshape(-1, self.n_timepoints)  # [batch_size * n_rois, timepoints]
        res_out = self.res_encoders[batch_idx](gin_out)  # [batch_size * n_rois, hidden_dim]
        residual = self.shared_residual(gin_out)  # [batch_size * n_rois, hidden_dim]

        # 3. 最终特征
        fused = res_out + residual  # [batch_size * n_rois, hidden_dim]

        return fused

    def forward(self, x):
        """
        Args:
            x: 视图列表，包含：
               - 3个节点掩码视图: [batch_size, n_rois, n_rois]
               - 3个边掩码视图: [batch_size, n_rois, n_rois]
               - 3个特征掩码视图: [batch_size, n_rois, timepoints]
        """
        batch_size = x[0].size(0)

        reconstructions = []
        mu_list = []
        logvar_list = []

        # 处理每组视图
        for i in range(0, len(x), 3):
            node_view = x[i]  # [batch_size, n_rois, n_rois]
            edge_view = x[i + 1]  # [batch_size, n_rois, n_rois]
            feat_view = x[i + 2]  # [batch_size, n_rois, timepoints]

            # 处理特征视图
            feat_flat = feat_view.reshape(batch_size * self.n_rois, -1)
            h = self.encode(feat_flat, i // 3, node_view, edge_view)
            h = h.reshape(batch_size, self.n_rois, -1)
            h = h.mean(dim=1)  # [batch_size, hidden_dim]

            # 计算mu和logvar
            mu = self.fc_mu[i // 3](h)  # [batch_size, latent_dim]
            logvar = self.fc_logvar[i // 3](h)  # [batch_size, latent_dim]

            mu_list.append(mu)
            logvar_list.append(logvar)

        # 计算平均的mu和logvar
        mu_mean = torch.stack(mu_list).mean(dim=0)
        logvar_mean = torch.stack(logvar_list).mean(dim=0)

        if self.training:
            # 添加小噪声以增加随机性
            noise = torch.randn_like(mu_mean) * 0.1
            mu_mean = mu_mean + noise

        z_fused = self.reparameterize(mu_mean, logvar_mean)

        # 使用融合的特征进行解码和分类
        for i in range(len(mu_list)):
            recon = self.decoder[i](z_fused)  # [batch_size, n_rois * n_timepoints]
            recon = recon.view(batch_size, self.n_rois, self.n_timepoints)
            reconstructions.append(recon)

        # 分类
        logits = self.classifier(z_fused)

        return reconstructions, mu_list, logvar_list, z_fused, logits

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def consistency_loss(self, reconstructions, mu_list, logvar_list, z_fused, logits, orig_views, labels):
        """计算一致性损失"""
        batch_size = reconstructions[0].size(0)

        # 1. 重建损失 - 结合MSE和余弦相似性
        recon_loss = 0
        for recon, orig in zip(reconstructions, orig_views[2::3]):
            # 重塑为 [batch_size, n_rois, timepoints]
            recon = recon.reshape(batch_size, self.n_rois, -1)
            orig = orig.reshape(batch_size, self.n_rois, -1)

            # MSE损失
            mse = F.mse_loss(recon, orig)

            # 余弦相似性损失
            recon_flat = recon.reshape(-1, recon.size(-1))
            orig_flat = orig.reshape(-1, orig.size(-1))
            cos_sim = 1 - F.cosine_similarity(recon_flat, orig_flat, dim=1).mean()

            # 组合两种损失
            combined_loss = 0.7 * mse + 0.3 * cos_sim
            recon_loss += combined_loss

        recon_loss = recon_loss / len(reconstructions)

        # 2. KL散度损失
        kl_loss = 0
        for mu, logvar in zip(mu_list, logvar_list):
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss += kl.mean()
        kl_loss = kl_loss / len(mu_list)

        # 3. 分类损失
        if labels is not None and logits is not None:
            cls_loss = F.cross_entropy(logits, labels)  # 移除class_weights参数
            current_acc = (logits.argmax(dim=1) == labels).float().mean()
        else:
            cls_loss = torch.tensor(0.0, device=logits.device)
            current_acc = 0.0

        # 添加L2正则化
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param)

        # 重新分配损失权重
        total_loss = (
                0.7 * recon_loss +
                0.3 * kl_loss +
                1 * cls_loss +
                0.001 * l2_reg  # 添加L2正则化损失
        )

        # 记录详细的损失信息
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'cls_loss': cls_loss.item(),
            'current_acc': current_acc.item() if isinstance(current_acc, torch.Tensor) else current_acc
        }

        return total_loss, loss_dict


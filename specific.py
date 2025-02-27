import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads=4, num_layers=2, dropout=0.4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 修改特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),  # 替换ReLU
            nn.Dropout(dropout)
        )

        # 简化Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # 减小前馈网络维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 修改输出投影层
        self.mu_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),  # 替换ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        self.logvar_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),  # 替换ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, x, node_view=None, edge_view=None):
        """
        Args:
            x: [batch_size, timepoints, n_rois] = [64, 137, 116]
            node_view: [batch_size, n_rois, n_rois] = [64, 116, 116]
            edge_view: [batch_size, n_rois, n_rois] = [64, 116, 116]
        """
        batch_size, timepoints, n_rois = x.size()
        x = x.contiguous()

        if node_view is not None and edge_view is not None:
            # 对每个时间点分别处理
            x_reshaped = x.transpose(1, 2)  # [64, 116, 137]

            # 直接使用视图信息
            node_message = torch.bmm(x_reshaped, node_view)  # [64, 116, 137]
            edge_message = torch.bmm(x_reshaped, edge_view)  # [64, 116, 137]

            # 简单平均
            x = (node_message + edge_message) / 2  # [64, 116, 137]
            x = x.transpose(1, 2)  # [64, 137, 116]

        # 对每个时间点进行特征投影
        x_flat = x.reshape(-1, n_rois)  # [64*137, 116]
        x = self.feature_proj(x_flat)  # [64*137, 128]
        x = x.reshape(batch_size, timepoints, self.hidden_dim)  # [64, 137, 128]

        # Transformer编码
        x = self.transformer(x)  # [64, 137, 128]

        # 添加线性层，将形状转换为 [64, 137, 116]
        linear_layer = nn.Linear(self.hidden_dim, self.n_rois)  # self.hidden_dim = 128, self.n_rois = 116
        linear_layer = linear_layer.to(x.device)  # 将线性层移到与x相同的设备上
        y = linear_layer(x)  # [64, 137, 116]

        # 计算皮尔逊相关系数
        def calculate_pearson_correlation(data):
            """计算皮尔逊相关系数矩阵（不进行阈值化）
            Args:
                data: 输入数据 [batch_size, ROIs, timepoints]
            Returns:
                相关系数矩阵 [batch_size, ROIs, ROIs]
            """
            batch_size, n_rois, timepoints = data.shape
            corr_matrices = []

            for i in range(batch_size):
                # 标准化数据
                x = data[i]  # [ROIs, timepoints]
                x = x - x.mean(dim=1, keepdim=True)
                x = x / (x.std(dim=1, keepdim=True) + 1e-8)  # 添加小值避免除零

                # 计算相关系数矩阵
                corr = torch.mm(x, x.t()) / (timepoints - 1)
                
                # 处理可能的数值不稳定性
                corr = torch.clamp(corr, min=-1.0, max=1.0)
                corr = torch.nan_to_num(corr, 0.0)
                
                corr_matrices.append(corr)

            return torch.stack(corr_matrices).to(data.device)

        # 计算皮尔逊相关系数
        pearson_corr = calculate_pearson_correlation(y.transpose(1, 2))  # [64, 116, 116]

        # 按batch取平均值
        mean_pearson_corr = pearson_corr.mean(dim=0)  # [116, 116]

        # 保存结果到txt文件
        output_file = 'pearson_correlation_results.txt'
        np.savetxt(output_file, mean_pearson_corr.detach().cpu().numpy(), fmt='%.4f')

        print(f"皮尔逊相关系数已保存到 {output_file}")

        # 全局平均池化
        x = x.mean(dim=1)  # [64, 128]

        # 生成分布参数
        mu = self.mu_proj(x)  # [64, 256]
        logvar = self.logvar_proj(x)  # [64, 256]

        return mu, logvar


class CLUB(nn.Module):
    def __init__(self, vcode_dim, ccode_dim, hidden_size):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(vcode_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, ccode_dim)
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(vcode_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, ccode_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        """计算条件概率的对数似然"""
        mu, logvar = self.get_mu_logvar(x_samples)
        return -0.5 * (torch.sum(
            (mu - y_samples) ** 2 / logvar.exp() + logvar + np.log(2 * np.pi),
            dim=1
        )).mean()

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # 正样本对的条件概率对数
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # 负样本对的条件概率对数
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def learning_loss(self, x_samples, y_samples):
        """计算用于优化的损失"""
        return -self.loglikeli(x_samples, y_samples)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        Args:
            alpha: 类别权重，用于处理类别不平衡
            gamma: 聚焦参数，用于调节难易样本的权重
            reduction: 损失计算方式，'mean', 'sum' 或 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 初始化 alpha
        if alpha is None:
            alpha = [1.0, 1.0]  # 默认每个类别权重相等
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits [N, C]
            targets: 目标标签 [N]
        """
        # 确保输入和目标在同一设备上
        targets = targets.to(inputs.device)
        self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = self.alpha[targets]

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiViewTransformerVAE(nn.Module):
    def __init__(self, input_shape, consistency_model, hidden_dim=64, dropout=0.5, weight_decay=0.01, num_views=9):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.weight_decay = weight_decay

        # 修改：正确获取维度信息
        if isinstance(input_shape, (list, tuple)):
            self.n_timepoints = input_shape[-2]  # 137 (时间点数量)
            self.n_rois = input_shape[-1]  # 116 (ROI数量)
        else:
            self.n_timepoints = input_shape.shape[-2]  # 137 (时间点数量)
            self.n_rois = input_shape.shape[-1]  # 116 (ROI数量)

        # 打印维度信息以便调试
        print(f"\nMultiViewTransformerVAE维度信息:")
        print(f"输入形状: {input_shape}")
        print(f"时间点数量: {self.n_timepoints}")  # 137
        print(f"ROIs数量: {self.n_rois}")  # 116

        # 获取consistency模型的latent_dim
        self.latent_dim = consistency_model.latent_dim  # 使用consistency模型的latent_dim(256)

        # 保存预训练的consistency模型
        self.consistency_model = consistency_model

        # 修改fusion_layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.latent_dim * 2, hidden_dim * 2),  # 512 -> 256 (256*2 -> 128*2)
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 256 -> 128
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, self.latent_dim)  # 128 -> 256
        )

        # 为每个视图创建独立的Transformer编码器
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                input_dim=self.n_rois,  # 使用 n_rois(116) 作为输入维度
                hidden_dim=hidden_dim,  # 128
                latent_dim=self.latent_dim,  # 256
                num_heads=4,
                num_layers=2,
                dropout=0.5
            ) for _ in range(num_views)
        ])

        # 修改CLUB估计器的初始化
        self.club = CLUB(
            vcode_dim=self.latent_dim,
            ccode_dim=self.latent_dim,
            hidden_size=hidden_dim * 2
        )

        # 修改分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 2)
        )

        # 2. 增加L2正则化强度
        self.l2_lambda = 0.01  # 可以尝试 0.01-0.05

        # 3. 添加特征正则化
        self.feature_dropout = nn.Dropout1d(0.2)  # 改用dropout1d

        # 修改解码器
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim * 2, hidden_dim * 4),  # 512 -> 256 (256*2 -> 128*4)
                nn.LayerNorm(hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, self.n_timepoints * self.n_rois),
                nn.Tanh()
            ) for _ in range(num_views)
        ])

        # 增加标签平滑系数
        self.label_smoothing = 0.2  # 可以尝试 0.1-0.2

    def reparameterize(self, mu, logvar):
        """重参数化采样"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, views, original_features):
        """
        Args:
            views: 9个视图的列表，每3个视图一组（node_view, edge_view, feat_view）
            original_features: 原始的特征数据 [batch_size, timepoints, n_rois]
        """
        # 添加dropout等随机性操作
        self.train()  # 即使在测试时也保持一定的随机性
        with torch.no_grad():
            reconstructions, mu_list, logvar_list, z_fused, logits = self.consistency_model(views)
            consistency_repr = z_fused  # 使用融合的潜在表示
        self.eval()

        mu_list = []
        logvar_list = []
        z_list = []
        fused_list = []
        mi_losses = []
        reconstructions = []
        recon_loss = 0

        # 每三个视图一组处理
        for i in range(0, len(views), 3):
            node_view = views[i]
            edge_view = views[i + 1]
            feat_view = views[i + 2]

            # 获取编码
            mu, logvar = self.encoders[i // 3](feat_view, node_view, edge_view)
            z = self.reparameterize(mu, logvar)

            # 融合一致性表示
            fused = self.fusion_layer(torch.cat([z, consistency_repr], dim=1))

            # 重建
            combined = torch.cat([z, consistency_repr], dim=1)
            recon = self.decoders[i // 3](combined)
            recon = recon.view(-1, self.n_timepoints, self.n_rois)  # [batch_size, 137, 116]

            # 添加重建结果到列表中
            reconstructions.append(recon)

            # 使用原始特征计算重建损失
            target = original_features  # 使用原始特征作为目标

            # 确保张量内存连续和维度一致
            recon = recon.contiguous()
            target = target.contiguous()

            # MSE损失
            mse_loss = F.mse_loss(recon, target)

            # L1损失
            l1_loss = F.l1_loss(recon, target)

            # 余弦相似度损失
            # 首先将张量展平为2D: [batch_size, timepoints*rois]
            recon_flat = recon.reshape(recon.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)
            # 计算余弦相似度 (1 - cos_sim 使得相似度越高损失越小)
            cos_loss = 1 - F.cosine_similarity(recon_flat, target_flat).mean()

            # 组合三种损失
            combined_loss = (0.6 * mse_loss +  # MSE损失权重
                             0.1 * l1_loss +  # L1损失权重
                             0.3 * cos_loss)  # 余弦相似度损失权重

            recon_loss += combined_loss

            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)
            fused_list.append(fused)

            # 只计算与一致性表示的互信息
            mi_loss = self.club.learning_loss(z, consistency_repr)
            mi_losses.append(mi_loss)

        # 计算重建损失
        recon_loss = recon_loss / len(reconstructions)

        # 4. 在forward过程中添加特征正则化
        features = self.feature_dropout(original_features)

        return {
            'mu_list': mu_list,
            'logvar_list': logvar_list,
            'z_list': z_list,
            'fused_list': fused_list,
            'consistency_repr': consistency_repr,
            'reconstructions': reconstructions,
            'logits': self.classifier(torch.cat([z_list[-1], consistency_repr], dim=1)),
            'mi_losses': mi_losses
        }

    def compute_loss(self, batch, views):
        """计算总损失"""
        features, labels = batch
        if features.shape[1] == self.n_rois:
            features = features.transpose(1, 2)

        outputs = self(views, features)

        # 1. 重建损失
        recon_loss = 0
        for recon in outputs['reconstructions']:
            target = features
            recon = recon.contiguous()
            target = target.contiguous()

            mse_loss = F.mse_loss(recon, target)
            l1_loss = F.l1_loss(recon, target)

            recon_flat = recon.reshape(recon.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)
            cos_loss = 1 - F.cosine_similarity(recon_flat, target_flat).mean()

            combined_loss = (0.6 * mse_loss +
                             0.1 * l1_loss +
                             0.3 * cos_loss)

            recon_loss += combined_loss

        recon_loss = recon_loss / len(outputs['reconstructions'])

        # 2. KL散度损失
        kl_loss = 0
        for mu, logvar in zip(outputs['mu_list'], outputs['logvar_list']):
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss += kl.mean()
        kl_loss = kl_loss / len(outputs['mu_list'])

        # 3. 互信息损失
        mi_loss = 0.0
        for i, z in enumerate(outputs['z_list']):
            mi_loss += 0.01 * self.club.learning_loss(z, outputs['consistency_repr'])
        mi_loss = mi_loss / len(outputs['z_list'])


        # 4. Focal Loss for 分类
        # 计算类别权重
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_counts = torch.zeros(2, dtype=torch.float32, device=labels.device)  # 修改：指定dtype为float32
        class_counts[unique_labels] = counts.float()  # 修改：将counts转换为float
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum()

        # 初始化Focal Loss
        focal_loss = FocalLoss(
            alpha=weights.tolist(),
            gamma=2.0,
            reduction='mean'
        )

        # 计算分类损失
        cls_loss = 0
        for i in range(0, len(views), 3):
            z = outputs['z_list'][i // 3]
            logits = self.classifier(torch.cat([z, outputs['consistency_repr']], dim=1))
            cls_loss += focal_loss(logits, labels)
        cls_loss = cls_loss / (len(views) // 3)

        # 收集所有视图的预测
        all_view_probs = []
        for i in range(0, len(views), 3):
            z = outputs['z_list'][i // 3]
            logits = self.classifier(torch.cat([z, outputs['consistency_repr']], dim=1))
            probs = F.softmax(logits, dim=1)
            all_view_probs.append(probs)

        # 计算平均概率
        avg_probs = torch.stack(all_view_probs).mean(dim=0)
        predictions = torch.argmax(avg_probs, dim=1)

        # 计算总体精度
        accuracy = (predictions == labels).float().mean() * 100

        # 计算每个类别的精度
        class_acc = []
        for cls in range(2):
            mask = labels == cls
            if mask.sum() > 0:
                class_acc.append((predictions[mask] == labels[mask]).float().mean())
            else:
                class_acc.append(torch.tensor(0.0, device=labels.device))

        # 3. 添加L2正则化损失
        l2_reg = torch.tensor(0., device=features.device)
        for param in self.classifier.parameters():
            l2_reg += torch.norm(param)

        # 6. 修改损失权重比例
        total_loss = (
                0.4 * recon_loss +  # 进一步降低重建损失权重
                0.2 * kl_loss +  # 降低KL散度权重
                0.1 * mi_loss +  # 保持互信息损失权重
                1.5 * cls_loss +  # 增加分类损失权重
                self.l2_lambda * l2_reg  # 增加L2正则化权重
        )

        # 7. 添加label smoothing
        if self.training:
            # 软化标签
            smooth_factor = self.label_smoothing
            n_classes = 2
            labels_one_hot = F.one_hot(labels, n_classes).float()
            labels_smooth = (1.0 - smooth_factor) * labels_one_hot + \
                            smooth_factor / n_classes

            cls_loss = -torch.sum(labels_smooth * F.log_softmax(outputs['logits'], dim=1), dim=1).mean()

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'mi_loss': mi_loss,
            'cls_loss': cls_loss.item(),
            'l2_loss': l2_reg.item(),
            'accuracy': accuracy.item(),
            'class_acc': [acc.item() for acc in class_acc],
            'logits': outputs['logits'],
            'predictions': predictions
        }


class SpecificEvaluator:
    def __init__(self, model, device):
        """
        初始化评估器
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_binary(self, data_loader, view_generator):
        """对二分类任务进行详细评估"""
        self.model.eval()
        if hasattr(view_generator, 'training'):
            view_generator.training = False

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                views = view_generator.generate_all_views(features)
                outputs = self.model(views, features)

                # 收集每个视图的预测概率
                view_probs = []
                for i in range(0, len(views), 3):
                    z = outputs['z_list'][i // 3]
                    logits = self.model.classifier(torch.cat([z, outputs['consistency_repr']], dim=1))
                    probs = F.softmax(logits, dim=1)
                    view_probs.append(probs)

                # 计算平均概率并获取预测
                avg_probs = torch.stack(view_probs).mean(dim=0)
                preds = avg_probs.argmax(dim=1)

                # 收集结果
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # 正确计算各项指标
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='weighted') * 100
        recall = recall_score(all_labels, all_preds, average='weighted') * 100
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100

        # 打印预测分布
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        print("\n预测分布:")
        for label, count in zip(unique_preds, pred_counts):
            print(f"类别 {label}: {count} ({count / len(all_preds) * 100:.2f}%)")

        # 打印混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print("\n混淆矩阵:")
        print(cm)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
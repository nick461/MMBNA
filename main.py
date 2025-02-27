import torch
import sys
from options import getargs
from dataset import BrainConnectivityDataset
from utils import MultiViewGenerator
from consistency import ViewConsistencyVAE
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import json
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.multiprocessing as mp
from collections import defaultdict
import csv
import torch.nn.functional as F
from datetime import timedelta
import socket
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


def save_checkpoint(model, optimizer, epoch, val_loss, args, config_name=None):
    """保存模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        val_loss: 验证损失
        args: 参数
        config_name: 配置名称（用于区分不同实验）
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 根据配置名称选择保存路径
    if config_name == 'original_loss':
        model_name = 'consistency_best_model_multiview.pt'
    elif config_name == 'ablation_single_view':
        model_name = 'consistency_best_model_singleview.pt'
    else:
        model_name = 'best_model.pt'

    checkpoint_path = os.path.join(args.save_dir, model_name)

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    print(f'Best model saved to {checkpoint_path} (loss: {val_loss:.4f})')


def save_experiment_results(args, fold_results, save_dir):
    """保存实验结果和配置信息
    Args:
        args: 实验参数
        fold_results: 每折的结果列表
        save_dir: 保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(save_dir, f'consistency_results_{timestamp}.txt')

    with open(result_file, 'w') as f:
        # 写入实验配置
        f.write("实验配置：\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型：Consistency VAE\n")
        f.write(f"学习率：{args.lr}\n")
        f.write(f"批次大小：{args.batch_size}\n")
        f.write(f"训练轮数：{args.epochs}\n")
        f.write(f"隐藏层维度：{args.hidden_dim}\n")
        f.write(f"潜在空间维度：{args.latent_dim}\n")
        f.write(f"视图数量：9\n\n")

        # 写入每折的结果
        f.write("各折验证结果：\n")
        f.write("=" * 50 + "\n")

        accuracies = []
        aucs = []
        f1s = []
        precisions = []
        recalls = []

        for fold, results in enumerate(fold_results):
            f.write(f"\n第 {fold + 1} 折：\n")
            f.write(f"├─ 准确率: {results['accuracy'] * 100:.2f}%\n")
            f.write(f"├─ AUC: {results['auc'] * 100:.2f}%\n")
            f.write(f"├─ F1分数: {results['f1'] * 100:.2f}%\n")
            f.write(f"├─ 精确率: {results['precision'] * 100:.2f}%\n")
            f.write(f"└─ 召回率: {results['recall'] * 100:.2f}%\n")

            accuracies.append(results['accuracy'])
            aucs.append(results['auc'])
            f1s.append(results['f1'])
            precisions.append(results['precision'])
            recalls.append(results['recall'])

        # 写入总体性能
        f.write("\n总体性能：\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%\n")
        f.write(f"AUC: {np.mean(aucs) * 100:.2f}% ± {np.std(aucs) * 100:.2f}%\n")
        f.write(f"F1分数: {np.mean(f1s) * 100:.2f}% ± {np.std(f1s) * 100:.2f}%\n")
        f.write(f"精确率: {np.mean(precisions) * 100:.2f}% ± {np.std(precisions) * 100:.2f}%\n")
        f.write(f"召回率: {np.mean(recalls) * 100:.2f}% ± {np.std(recalls) * 100:.2f}%\n")

        print(f"\n实验结果已保存到: {result_file}")


def load_data(args):
    """加载数据集"""
    data_path = args.data_path + ".mat"

    try:
        data = sio.loadmat(data_path, verify_compressed_data_integrity=False)

        if args.dataset == 3:  # ADNI
            features = torch.from_numpy(data['timeseries'].astype(np.float32))
            features = features.permute(0, 2, 1)
            labels = torch.from_numpy(data['label'].squeeze().astype(np.int64))

            # 打印原始标签分布
            print("\n原始标签分布:")
            unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"类别 {label}: {count} 样本")

            # 将标签转换为二分类 (>1)
            labels = (labels > 1).long()

            print("\n转换后的二分类标签分布:")
            cn_count = torch.sum(labels == 0).item()
            other_count = torch.sum(labels == 1).item()
            print(f"类别 0 (CN): {cn_count} 样本 (类别1: {154}个)")
            print(f"类别 1 (SMC/MCI/AD): {other_count} 样本 (类别2: {165}个 + 类别3: {145}个 + 类别4: {99}个)")

        # 数据标准化
        if args.normalize_data:
            features = (features - features.mean(dim=-1, keepdim=True)) / (features.std(dim=-1, keepdim=True) + 1e-8)

        # 创建数据集
        dataset = BrainConnectivityDataset(features, labels)
        return dataset

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise


def get_experiment_config():
    """定义不同的实验配置"""
    configs = {
        'original_loss': {
            'epochs': 60,
            'lr': 0.001,
            'hidden_dim': 137,
            'latent_dim': 256,
            'beta': 1e-8,
            'gamma': 0.05,
            'description': '原始损失函数(MSE+图结相似性)',
            'use_scheduler': True,
            'dropout': 0.4,
            'weight_decay': 1e-4,
            'cls_weight': 2.0,
            'patience': 30,
            'loss_type': 'original'
        },
        'ablation_single_view': {  # 添加单视图消融实验配置
            'epochs': 60,
            'lr': 0.001,
            'hidden_dim': 137,
            'latent_dim': 256,
            'beta': 1e-8,
            'gamma': 0.05,
            'description': '单视图消融实验（无多视图构造）',
            'use_scheduler': True,
            'dropout': 0.4,
            'weight_decay': 1e-4,
            'cls_weight': 2.0,
            'patience': 30,
            'loss_type': 'original',
            'ablation_single_view': True  # 添加标志位
        }
    }
    return configs


def train_epoch(model, train_loader, optimizer, device, epoch, view_generator, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loss_dict_sum = defaultdict(float)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (features, labels) in enumerate(pbar):
        # features shape: [batch_size, 116, 137]  # 直接使用原始特征形状
        features = features.to(device)
        labels = labels.to(device)

        if config.get('ablation_single_view', False):
            # 单视图模式：直接计算相关性矩阵，不需要转置
            x_normalized = (features - features.mean(dim=2, keepdim=True)) / (features.std(dim=2, keepdim=True) + 1e-8)
            pc_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # [batch_size, 116, 116]

            # 构造视图列表
            views = []
            for _ in range(3):
                views.extend([
                    pc_matrix,  # 节点视图 [batch_size, 116, 116]
                    pc_matrix,  # 边视图 [batch_size, 116, 116]
                    features  # 特征视图 [batch_size, 116, 137]
                ])
        else:
            views = view_generator.generate_all_views(features,
                                                      ablation_single_view=config.get('ablation_single_view', False))

        # 前向传播
        reconstructions, mu_list, logvar_list, z_fused, logits = model(views)

        # 计算损失
        loss, loss_dict = model.consistency_loss(
            reconstructions=reconstructions,
            mu_list=mu_list,
            logvar_list=logvar_list,
            z_fused=z_fused,
            logits=logits,
            orig_views=views,
            labels=labels
        )

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计信息
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # 更新损失字典
        for k, v in loss_dict.items():
            loss_dict_sum[k] += v

        # 更新进度条，显示详细的损失组成
        batch_acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'recon': f'{loss_dict_sum["recon_loss"] / (batch_idx + 1):.4f}',
            'kl': f'{loss_dict_sum["kl_loss"] / (batch_idx + 1):.4f}',
            'cls': f'{loss_dict_sum["cls_loss"] / (batch_idx + 1):.4f}',
            'l2': f'{loss_dict_sum["l2_reg"] / (batch_idx + 1):.4f}',
            'acc': f'{batch_acc:.1f}%'
        })

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total

    # 计算平均损失字典
    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}

    # 在epoch结束时打印更详细的损失组成
    print(f'\nEpoch {epoch} 损失组成详情:')
    print('=' * 50)
    print(f'总损失: {avg_loss:.4f}')
    for loss_name, loss_value in avg_loss_dict.items():
        print(f'{loss_name}: {loss_value:.4f} ({(loss_value / avg_loss) * 100:.1f}%)')
    print(f'训练准确率: {avg_acc:.1f}%')
    print('=' * 50)

    return avg_loss, avg_acc, avg_loss_dict, batch_acc


def validate_epoch(model, val_loader, device, args, view_generator):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            # 生成视图 - 这里也只接收views
            views = view_generator.generate_all_views(features)
            views = [view.to(device) for view in views]

            # 前向传播
            reconstructions, mu_list, logvar_list, z_fused, logits = model(views)

            # 计算损失
            loss, _ = model.consistency_loss(
                views, reconstructions, mu_list, logvar_list, labels, logits
            )

            # 收集预测结果
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 计算各种指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds)
    }

    val_loss = total_loss / len(val_loader)

    # 在验证结束时打印详细的损失组成
    print(f'\n验证集评估结果:')
    print('=' * 50)
    print(f'验证损失: {val_loss:.4f}')
    print(f'准确率: {metrics["accuracy"] * 100:.2f}%')
    print(f'AUC: {metrics["auc"] * 100:.2f}%')
    print(f'F1分数: {metrics["f1"] * 100:.2f}%')
    print(f'精确率: {metrics["precision"] * 100:.2f}%')
    print(f'召回率: {metrics["recall"] * 100:.2f}%')
    print('=' * 50)

    return val_loss, metrics


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, args, view_generator, config):
    """训练模型"""
    # 初始化早停相关变量
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0  # 计数器，记录验证损失没有改善的轮数
    patience = config['patience']  # 从配置中获取耐心值，这里是15

    # 添加准确率列表来记录每个epoch的准确率
    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        # 添加训练进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        # 训练阶段
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 生成多视图
            views = view_generator.generate_all_views(features, config.get('ablation_single_view', False))

            # 前向传播
            reconstructions, mu_list, logvar_list, z_fused, logits = model(views)

            # 计算损失
            loss, loss_dict = model.consistency_loss(reconstructions, mu_list, logvar_list, z_fused, logits, views,
                                                     labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算准确率
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item() * 100:.2f}%',
                'recon': f'{loss_dict["recon_loss"]:.4f}',
                'kl': f'{loss_dict["kl_loss"]:.4f}',
                'cls': f'{loss_dict["cls_loss"]:.4f}'
            })

        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches * 100
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_acc = 0
        num_val_batches = 0
        all_preds = []
        all_labels = []

        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc='Validating')

        with torch.no_grad():
            for features, labels in val_pbar:
                features = features.to(device)
                labels = labels.to(device)

                views = view_generator.generate_all_views(features, config.get('ablation_single_view', False))
                reconstructions, mu_list, logvar_list, z_fused, logits = model(views)

                loss, _ = model.consistency_loss(reconstructions, mu_list, logvar_list, z_fused, logits, views, labels)

                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()
                num_val_batches += 1

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新验证进度条信息
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc.item() * 100:.2f}%'
                })

        val_loss = val_loss / num_val_batches
        val_acc = val_acc / num_val_batches * 100
        val_accuracies.append(val_acc)

        # 计算其他指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }

        # 计算 AUC（仅适用于二分类）
        if len(np.unique(all_labels)) == 2:
            metrics['auc'] = roc_auc_score(all_labels, all_preds)
        else:
            metrics['auc'] = 0

        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_loss)

        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = metrics
            counter = 0  # 重置计数器

            # 保存最佳模型
            if hasattr(args, 'save_dir'):
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    args,
                    config_name=config.get('config_name', None)
                )
        else:
            counter += 1  # 验证损失没有改善，计数器加1

        # 早停检查
        if counter >= patience:  # 如果连续patience轮验证损失都没有改善
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break  # 停止训练

    return best_val_loss, best_epoch, best_metrics


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_process(args):
    """训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = load_data(args)
    input_shape = dataset[0][0].shape
    dataset_size = len(dataset)

    # 1. 首先划分出测试集 (20%)
    indices = list(range(dataset_size))
    test_split = int(np.floor(0.2 * dataset_size))

    # 设置随机种子确保可重复性
    np.random.seed(42)
    np.random.shuffle(indices)
    test_indices = indices[:test_split]
    remaining_indices = indices[test_split:]

    # 保存测试集索引
    test_indices_path = os.path.join(args.save_dir, 'test_indices.npy')
    np.save(test_indices_path, test_indices)

    # 2. 在剩余数据上进行K折交叉验证
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    print(f"\n数据集划分:")
    print(f"总样本数: {dataset_size}")
    print(f"测试集大小: {len(test_indices)}")
    print(f"用于交叉验证的样本数: {len(remaining_indices)}")
    print(f"每折训练集大小: ~{len(remaining_indices) * (k_folds - 1) // k_folds}")
    print(f"每折验证集大小: ~{len(remaining_indices) // k_folds}")
    print(f"输入形状: {input_shape}")

    # 创建视图生成器
    view_generator = MultiViewGenerator(
        node_mask_ratio=args.node_mask_ratio,
        edge_mask_ratio=args.edge_mask_ratio,
        feature_mask_ratio=args.feature_mask_ratio
    ).to(device)

    # 获取实验配置
    configs = get_experiment_config()

    # 对每个配置进行训练
    for config_name, config in configs.items():
        config['config_name'] = config_name

        print("\n" + "=" * 50)
        print(f"开始实验配置: {config_name}")
        print("=" * 50)
        print(f"实验描述: {config['description']}")
        print("\n主要超参数:")
        print(f"├─ 学习率: {config['lr']}")
        print(f"├─ 批次大小: {args.batch_size}")
        print(f"├─ 最大轮次: {config['epochs']}")
        print(f"├─ 早停耐心值: {config['patience']}")
        print(f"├─ 隐藏层维度: {config['hidden_dim']}")
        print(f"├─ 潜在空间维度: {config['latent_dim']}")
        print(f"├─ Beta (VAE权重): {config['beta']}")
        print(f"├─ Gamma (一致性权重): {config['gamma']}")
        print(f"├─ Dropout率: {config['dropout']}")
        print(f"├─ 权重衰减: {config['weight_decay']}")
        print(f"└─ 分类权重: {config['cls_weight']}")
        print("=" * 50 + "\n")

        # 在剩余数据上进行K折交叉验证
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_indices)):
            # 获取实际的训练和验证索引
            train_indices = [remaining_indices[i] for i in train_idx]
            val_indices = [remaining_indices[i] for i in val_idx]

            print(f"\n第 {fold + 1} 折:")
            print(f"训练集大小: {len(train_indices)}")
            print(f"验证集大小: {len(val_indices)}")

            # 创建训练集和验证集
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)

            # 创建数据加载器
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # 创建模型
            model = ViewConsistencyVAE(
                input_shape=input_shape,
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                beta=config['beta'],
                gamma=config['gamma'],
                num_classes=args.num_classes,
                dropout=0.5
            ).to(device)

            # 创建优化器和调度器
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['lr'],
                weight_decay=1e-4
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            ) if config.get('use_scheduler', False) else None

            # 计算类别权重
            def calculate_class_weights(dataset):
                labels = [data[1] for data in dataset]
                unique_labels, counts = np.unique(labels, return_counts=True)
                weights = 1.0 / counts
                weights = weights / weights.sum()
                return torch.FloatTensor(weights).to(device)

            # 使用Focal Loss替换CrossEntropyLoss
            class_weights = calculate_class_weights(train_subset)
            criterion = FocalLoss(
                alpha=class_weights,  # 类别权重
                gamma=2.0,  # focal loss的聚焦参数，越大越关注难分类的样本
                reduction='mean'
            )

            # 训练当前折
            best_val_loss, best_epoch, best_metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
                view_generator=view_generator,
                config=config
            )

            # 保存当前折的结果
            fold_results.append({
                'fold': fold + 1,
                'accuracy': best_metrics['accuracy'],
                'auc': best_metrics['auc'],
                'f1': best_metrics['f1'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            })

            print(f"\n第 {fold + 1} 折训练完成:")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            print(f"最佳轮次: {best_epoch}")
            print(f"最佳准确率: {best_metrics['accuracy'] * 100:.2f}%")

            # 清理内存
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 保存实验结果
        save_experiment_results(args, fold_results, args.save_dir)


def main():
    args = getargs()

    if torch.cuda.is_available():
        print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n使用CPU训练")

    train_process(args)


if __name__ == "__main__":
    main()


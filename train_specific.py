import torch
import torch.nn as nn
from options import getargs
from dataset import BrainConnectivityDataset
from utils import MultiViewGenerator
from consistency import ViewConsistencyVAE
from specific import MultiViewTransformerVAE, SpecificEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os
import numpy as np
from tqdm import tqdm
from main import load_data
import datetime
import copy
from collections import defaultdict
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


def load_pretrained_consistency(args, device):
    """加载预训练的consistency模型"""
    # 检查多个可能的路径
    possible_paths = [
        os.path.join('checkpoints', 'consistency_best_model_multiview.pt')
    ]

    # 找到第一个存在的路径
    pretrained_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pretrained_path = path
            break

    if pretrained_path is None:
        raise FileNotFoundError("找不到预训练模型文件")

    try:
        # 加载预训练模型文件
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                                  dict) and 'model_state_dict' in checkpoint else checkpoint

        # 从权重维度推断参数
        hidden_dim = state_dict['fusion_encoder.0.bias'].shape[0]  # 137

        # 创建consistency模型
        dataset = load_data(args)
        input_shape = dataset[0][0].shape

        consistency_model = ViewConsistencyVAE(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            latent_dim=256,  # 使用512
            num_views=9,
            beta=1e-6,
            gamma=0.1,
            num_gin_layers=3,
            num_classes=2
        ).to(device)

        # 加载权重
        consistency_model.load_state_dict(state_dict)
        print("成功加载预训练模型")

        return consistency_model

    except Exception as e:
        print(f"\n加载预训练模型时发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise


def validate(model, val_loader, view_generator, device):
    """验证函数
    Args:
        model: 模型
        val_loader: 验证集数据加载器
        view_generator: 视图生成器
        device: 计算设备
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    # 添加验证进度条
    val_pbar = tqdm(val_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for features, labels in val_pbar:
            features = features.to(device)
            labels = labels.to(device)

            views = view_generator.generate_all_views(features)
            loss_dict = model.compute_loss((features, labels), views)

            total_loss += loss_dict['total_loss'].item()
            total_acc += loss_dict['accuracy']
            n_batches += 1

            # 更新进度条
            val_pbar.set_postfix({
                'loss': f'{total_loss / n_batches:.4f}',
                'acc': f'{total_acc / n_batches:.2f}%'
            })

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return avg_loss, avg_acc


def train_specific_process(args):
    """训练过程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载数据集
    dataset = load_data(args)
    
    # 加载之前保存的测试集索引
    test_indices_path = os.path.join(args.save_dir, 'test_indices.npy')
    if not os.path.exists(test_indices_path):
        raise FileNotFoundError("找不到测试集索引文件，请先运行主模型训练")
        
    test_indices = np.load(test_indices_path)
    all_indices = set(range(len(dataset)))
    train_val_indices = list(all_indices - set(test_indices))
    
    print("\n数据集划分:")
    print(f"├─ 总样本数: {len(dataset)}")
    print(f"├─ 训练验证集: {len(train_val_indices)} 样本")
    print(f"└─ 测试集: {len(test_indices)} 样本")
    
    # 创建训练验证集和测试集
    train_val_dataset = torch.utils.data.Subset(dataset, train_val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建5折交叉验证
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    # 记录每折的最佳结果
    fold_results = []
    best_overall_val_loss = float('inf')
    best_overall_model = None
    best_fold = -1

    # 对每一折进行训练
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
        print(f"\n{'=' * 50}")
        print(f"开始第 {fold + 1} 折训练")
        print(f"├─ 训练集样本数: {len(train_idx)}")
        print(f"└─ 验证集样本数: {len(val_idx)}")
        print(f"{'=' * 50}")

        # 创建数据加载器
        train_loader = DataLoader(
            torch.utils.data.Subset(train_val_dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            torch.utils.data.Subset(train_val_dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 创建视图生成器
        view_generator = MultiViewGenerator(
            node_mask_ratio=args.node_mask_ratio,
            edge_mask_ratio=args.edge_mask_ratio,
            feature_mask_ratio=args.feature_mask_ratio
        )

        # 加载预训练的一致性模型
        consistency_model = load_pretrained_consistency(args, device)

        # 创建specific模型
        specific_model = MultiViewTransformerVAE(
            input_shape=dataset[0][0].shape,
            consistency_model=consistency_model,
            hidden_dim=64,
            dropout=0.5,
            weight_decay=0.01
        ).to(device)

        # 优化器
        optimizer = torch.optim.AdamW(
            specific_model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )

        # 当前折的训练
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(args.epochs):
            # 训练阶段
            specific_model.train()
            train_loss = 0
            train_acc = 0
            n_batches = 0

            # 记录各个损失
            total_recon_loss = 0
            total_kl_loss = 0
            total_mi_loss = 0
            total_cls_loss = 0
            total_l2_loss = 0

            train_pbar = tqdm(train_loader, desc=f'Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs}')

            for features, labels in train_pbar:
                features = features.to(device)
                labels = labels.to(device)

                views = view_generator.generate_all_views(features)
                loss_dict = specific_model.compute_loss((features, labels), views)

                # 累加各个损失
                total_recon_loss += loss_dict['recon_loss']
                total_kl_loss += loss_dict['kl_loss']
                total_mi_loss += loss_dict['mi_loss']
                total_cls_loss += loss_dict['cls_loss']
                total_l2_loss += loss_dict['l2_loss']

                loss = loss_dict['total_loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += loss_dict['accuracy']
                n_batches += 1

                # 更新进度条，显示各损失的平均值
                train_pbar.set_postfix({
                    'loss': f'{train_loss / n_batches:.4f}',
                    'acc': f'{train_acc / n_batches:.2f}%',
                    'recon': f'{total_recon_loss / n_batches:.4f}',
                    'kl': f'{total_kl_loss / n_batches:.4f}',
                    'cls': f'{total_cls_loss / n_batches:.4f}',
                    'l2': f'{total_l2_loss / n_batches:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            avg_train_loss = train_loss / n_batches
            avg_train_acc = train_acc / n_batches

            # 验证阶段
            val_loss, val_acc = validate(specific_model, val_loader, view_generator, device)

            # 更新学习率
            scheduler.step(val_loss)

            # 保存当前折的最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(specific_model.state_dict())

            # 打印训练信息
            print(f'\nFold {fold + 1}, Epoch {epoch + 1} Summary:')
            print(f'├─ Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
            print(f'├─ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'└─ Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # 每个epoch结束时打印损失组成
            print(f'\nEpoch {epoch + 1} 损失组成:')
            print(f'├─ 重建损失: {total_recon_loss / n_batches:.4f}')
            print(f'├─ KL损失: {total_kl_loss / n_batches:.4f}')
            print(f'├─ 互信息损失: {total_mi_loss / n_batches:.4f}')
            print(f'├─ 分类损失: {total_cls_loss / n_batches:.4f}')
            print(f'└─ L2正则化: {total_l2_loss / n_batches:.4f}')

        # 记录当前折的结果
        fold_results.append({
            'fold': fold,
            'best_val_loss': best_val_loss,
            'model_state': best_model_state
        })

        # 更新全局最佳模型
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_model = best_model_state
            best_fold = fold

    # 打印所有折的结果
    print("\n交叉验证结果汇总:")
    for result in fold_results:
        print(f"第 {result['fold'] + 1} 折 - 最佳验证损失: {result['best_val_loss']:.4f}")

    print(f"\n最佳模型来自第 {best_fold + 1} 折")
    print(f"最佳验证损失: {best_overall_val_loss:.4f}")

    # 保存最佳模型
    save_path = os.path.join(args.save_dir, 'specific_best_model.pt')
    torch.save({
        'model_state_dict': best_overall_model,
        'best_fold': best_fold,
        'best_val_loss': best_overall_val_loss,
        'args': args
    }, save_path)
    print(f"\n保存最佳模型到: {save_path}")


def save_experiment_results(args, fold_results, save_dir):
    """保存实验结果和配置信息"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(save_dir, f'experiment_results_{timestamp}.txt')

    with open(result_file, 'w') as f:
        # 实验配置
        f.write("实验配置：\n")
        f.write(f"学习率：{args.lr}\n")
        f.write(f"批次大小：{args.batch_size}\n")
        f.write(f"训练轮数：{args.epochs}\n\n")

        # 各折验证结果
        f.write("各折验证结果：\n")
        for fold, results in enumerate(fold_results):
            f.write(f"\n第 {fold + 1} 折：\n")
            f.write("-" * 30 + "\n")

            # 获取分类报告
            report = results['classification_report']

            # 写入详细指标
            f.write("分类报告:\n")
            for label in ['Normal', 'Abnormal']:
                metrics = report[label]
                f.write(f"{label}类别:\n")
                f.write(f"├─ 精确率: {metrics['precision'] * 100:.2f}%\n")
                f.write(f"├─ 召回率: {metrics['recall'] * 100:.2f}%\n")
                f.write(f"└─ F1分数: {metrics['f1-score'] * 100:.2f}%\n")

            f.write(f"\n总体准确率: {report['accuracy'] * 100:.2f}%\n")
            f.write(f"ROC AUC: {results['roc_auc'] * 100:.2f}%\n")
            f.write(f"平均精确率: {results['average_precision'] * 100:.2f}%\n")

    print(f"\n实验结果已保存到: {result_file}")


def main():
    args = getargs()

    if torch.cuda.is_available():
        print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n使用CPU训练")

    train_specific_process(args)


if __name__ == "__main__":
    main()
import torch
import numpy as np
from options import getargs
from dataset import BrainConnectivityDataset
from consistency import ViewConsistencyVAE
from utils import MultiViewGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import os
from tqdm import tqdm
import scipy.io as sio
import torch.nn as nn
from datetime import datetime


def test_model(model, test_loader, device, view_generator, config):
    """测试模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="测试中"):
            features = features.to(device)
            labels = labels.to(device)
            
            if config.get('ablation_single_view', False):
                # 单视图模式
                features = features.transpose(1, 2)
                x_normalized = (features - features.mean(dim=2, keepdim=True)) / (features.std(dim=2, keepdim=True) + 1e-8)
                pc_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
                features = features.transpose(1, 2)
                
                views = []
                for _ in range(3):
                    views.extend([pc_matrix, pc_matrix, features])
            else:
                # 多视图模式
                views = view_generator.generate_all_views(features)
            
            # 前向传播
            reconstructions, mu_list, logvar_list, z_fused, logits = model(views)
            
            # 获取预测和概率
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算混淆矩阵元素
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0,  # 敏感性
        'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0   # 特异性
    }
    
    # 只在二分类问题中计算AUC
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    
    if n_classes == 2:
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        except Exception as e:
            print(f"警告：计算AUC时出错: {str(e)}")
            metrics['auc'] = 0.0
    else:
        print(f"警告：数据集包含 {n_classes} 个类别，跳过AUC计算")
        metrics['auc'] = 0.0
    
    return metrics


def save_test_results(metrics, config_name, save_dir):
    """保存测试结果到文本文件
    Args:
        metrics: 测试指标字典
        config_name: 配置名称（'multiview' 或 'singleview'）
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(save_dir, f'test_results_{config_name}_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"{config_name}模型测试结果:\n")
        f.write("="*50 + "\n\n")
        
        # 写入测试指标
        f.write("性能指标:\n")
        f.write("-"*30 + "\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']*100:.2f}%\n")
        f.write(f"敏感性 (Sensitivity): {metrics['sensitivity']*100:.2f}%\n")
        f.write(f"特异性 (Specificity): {metrics['specificity']*100:.2f}%\n")
        f.write(f"F1分数 (F1-Score): {metrics['f1']*100:.2f}%\n")
        f.write(f"精确率 (Precision): {metrics['precision']*100:.2f}%\n")
        f.write(f"召回率 (Recall): {metrics['recall']*100:.2f}%\n")
        if 'auc' in metrics:
            f.write(f"AUC: {metrics['auc']*100:.2f}%\n")
        
        # 写入时间戳
        f.write("\n测试时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    print(f"\n测试结果已保存到: {result_file}")


def main():
    args = getargs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_path = args.data_path + ".mat"
    try:
        data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        
        # 加载ADNI数据集
        features = torch.from_numpy(data['timeseries'].astype(np.float32))
        features = features.permute(0, 2, 1)
        labels = torch.from_numpy(data['label'].squeeze().astype(np.int64))
        
        # 打印原始标签分布
        print("\n原始标签分布:")
        unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label}: {count} 样本")
        
        # 转换为二分类：CN vs MCI/AD
        labels = (labels > 1).long()  # CN/SMC vs EMCI/LMCI/AD
        
        print("\n转换后的二分类标签分布:")
        print("类别 0 (CN/SMC): ", torch.sum(labels == 0).item(), "样本")
        print("类别 1 (MCI/AD): ", torch.sum(labels == 1).item(), "样本")
        
        # 数据标准化
        if args.normalize_data:
            features = (features - features.mean(dim=-1, keepdim=True)) / (features.std(dim=-1, keepdim=True) + 1e-8)
        
        # 创建数据集
        dataset = BrainConnectivityDataset(features, labels)
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise
    
    # 加载测试集索引
    test_indices_path = os.path.join(args.save_dir, 'test_indices.npy')
    if not os.path.exists(test_indices_path):
        print(f"错误：找不到测试集索引文件 {test_indices_path}")
        print("请先运行训练脚本生成测试集索引")
        return
    
    test_indices = np.load(test_indices_path)
    
    # 验证测试集索引的有效性
    if max(test_indices) >= len(dataset):
        print(f"错误：测试集索引超出数据集范围")
        print(f"最大索引值: {max(test_indices)}")
        print(f"数据集大小: {len(dataset)}")
        return
    
    # 创建测试集
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"\n测试集信息:")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"测试集索引范围: [{min(test_indices)}, {max(test_indices)}]")
    
    # 创建测试集加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建视图生成器
    view_generator = MultiViewGenerator(
        node_mask_ratio=args.node_mask_ratio,
        edge_mask_ratio=args.edge_mask_ratio,
        feature_mask_ratio=args.feature_mask_ratio
    ).to(device)
    
    # 测试多视图模型
    multiview_model = ViewConsistencyVAE(
        input_shape=dataset.features[0].shape,
        hidden_dim=137,
        latent_dim=256,
        num_classes=2
    ).to(device)
    
    multiview_path = os.path.join(args.save_dir, 'consistency_best_model_multiview.pt')
    if not os.path.exists(multiview_path):
        print(f"错误：找不到多视图模型文件 {multiview_path}")
        return
    
    multiview_model.load_state_dict(torch.load(multiview_path)['model_state_dict'])
    
    # 添加类别权重
    class_weights = torch.FloatTensor([
        1.0 / torch.sum(labels == 0),
        1.0 / torch.sum(labels == 1)
    ]).to(device)
    
    # 在损失函数中使用类别权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    multiview_metrics = test_model(
        multiview_model, 
        test_loader, 
        device, 
        view_generator,
        {'ablation_single_view': False}
    )
    
    # 保存多视图测试结果
    save_test_results(multiview_metrics, 'multiview', args.save_dir)
    
    # 测试单视图模型
    singleview_model = ViewConsistencyVAE(
        input_shape=dataset.features[0].shape,
        hidden_dim=137,
        latent_dim=256,
        num_classes=2
    ).to(device)
    
    singleview_path = os.path.join(args.save_dir, 'consistency_best_model_singleview.pt')
    if not os.path.exists(singleview_path):
        print(f"错误：找不到单视图模型文件 {singleview_path}")
        return
    
    singleview_model.load_state_dict(torch.load(singleview_path)['model_state_dict'])
    
    singleview_metrics = test_model(
        singleview_model, 
        test_loader, 
        device, 
        view_generator,
        {'ablation_single_view': True}
    )
    
    # 保存单视图测试结果
    save_test_results(singleview_metrics, 'singleview', args.save_dir)
    
    # 打印结果
    print("\n多视图模型测试结果:")
    for k, v in multiview_metrics.items():
        print(f"{k}: {v*100:.2f}%")
    
    print("\n单视图模型测试结果:")
    for k, v in singleview_metrics.items():
        print(f"{k}: {v*100:.2f}%")


if __name__ == "__main__":
    main() 
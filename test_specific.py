import torch
import torch.nn as nn
from options import getargs
from dataset import Load_Data
from utils import MultiViewGenerator
from consistency import ViewConsistencyVAE
from specific import MultiViewTransformerVAE, SpecificEvaluator
from torch.utils.data import DataLoader
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as sio
from train_specific import load_pretrained_consistency
from main import load_data


def evaluate_model(model, data_loader, view_generator, device):
    """独立的模型评估函数
    Args:
        model: 待评估的模型
        data_loader: 数据加载器
        view_generator: 视图生成器
        device: 计算设备
    """
    model.eval()
    view_generator.eval()  # 设置视图生成器为评估模式

    all_labels = []
    all_preds = []
    all_probs = []  # 添加存储预测概率

    with torch.no_grad():
        for batch in data_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # 生成视图
            views = view_generator.generate_all_views(features)

            # 获取模型输出
            outputs = model(views, features)
            logits = outputs['logits']
            
            # 获取预测概率和类别
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 保存正类的概率

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算混淆矩阵元素
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    # 计算各项指标
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算AUC
    auc = roc_auc_score(all_labels, all_probs)

    # 打印详细信息
    print("\n评估指标详情:")
    print(f"准确率 (ACC): {accuracy*100:.2f}%")
    print(f"敏感性 (SEN): {sensitivity*100:.2f}%")
    print(f"特异性 (SPE): {specificity*100:.2f}%")
    print(f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall): {recall*100:.2f}%")
    print(f"F1分数: {f1*100:.2f}%")
    print(f"AUC: {auc*100:.2f}%")

    return {
        'accuracy': accuracy * 100,
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'auc': auc * 100,
        'confusion_matrix': np.array([[TN, FP], [FN, TP]]),
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs  # 保存预测概率
    }


def test_model(args):
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 加载数据集
    dataset = load_data(args)

    # 加载测试集索引
    test_indices_path = os.path.join(args.save_dir, 'test_indices.npy')
    test_indices = np.load(test_indices_path)
    print(f"\n加载测试集索引: {test_indices_path}")
    print(f"测试集大小: {len(test_indices)}")

    # 创建测试集数据加载器
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,  # 测试时不需要打乱数据
        num_workers=4,
        pin_memory=True
    )

    # 创建视图生成器
    view_generator = MultiViewGenerator(
        node_mask_ratio=args.node_mask_ratio,
        edge_mask_ratio=args.edge_mask_ratio,
        feature_mask_ratio=args.feature_mask_ratio
    )

    # 加载模型
    model_path = os.path.join(args.save_dir, 'specific_best_model.pt')
    checkpoint = torch.load(model_path)

    # 验证模型权重是否正确加载
    print("\n模型参数示例:")
    specific_model = MultiViewTransformerVAE(
        input_shape=dataset[0][0].shape,
        consistency_model=load_pretrained_consistency(args, device),
        hidden_dim=64,
        num_views=9
    ).to(device)
    original_params = list(specific_model.parameters())[0].clone()
    specific_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_params = list(specific_model.parameters())[0].clone()
    print(f"参数是否改变: {not torch.allclose(original_params, loaded_params)}")

    # 进行一次简单的前向传播测试
    specific_model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        test_features, test_labels = test_batch
        test_features = test_features.to(device)

        # 生成视图
        test_views = view_generator.generate_all_views(test_features)

        # 前向传播
        test_outputs = specific_model(test_views, test_features)
        print("\n测试前向传播:")
        print(f"输出logits: {test_outputs['logits'][:2]}")

    # 然后再进行完整的评估
    results = evaluate_model(specific_model, test_loader, view_generator, device)

    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"F1-score: {results['f1_score']:.2f}%")

    # 保存评估结果
    save_test_results(args, results)


def save_test_results(args, results):
    """保存测试结果"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.save_dir, f'test_results_{timestamp}.txt')

    with open(result_file, 'w') as f:
        f.write("测试结果\n")
        f.write("=" * 50 + "\n\n")

        # 只输出总体指标
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Precision: {results['precision']:.2f}%\n")
        f.write(f"Recall: {results['recall']:.2f}%\n")
        f.write(f"F1-score: {results['f1_score']:.2f}%\n")

    print(f"\n结果已保存到: {result_file}")


def write_metrics(f, metrics):
    """写入评估指标"""
    f.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
    for label in ['Normal', 'Abnormal']:
        f.write(f"\n{label}类别:\n")
        f.write(f"├─ 精确率: {metrics[label]['precision'] * 100:.2f}%\n")
        f.write(f"├─ 召回率: {metrics[label]['recall'] * 100:.2f}%\n")
        f.write(f"└─ F1分数: {metrics[label]['f1-score'] * 100:.2f}%\n")


if __name__ == "__main__":
    args = getargs()
    test_model(args) 
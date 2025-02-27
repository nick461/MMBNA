import scipy.io as sio
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import tensor, float32
from random import shuffle
import numpy as np
from torch.utils.data import Dataset
from options import getargs

args = getargs()


def dataread(train=True):
    """读取数据集"""
    data_path = args.data_path + ".mat"
    mat_data = sio.loadmat(data_path)
    data = []

    if args.dataset == 1:  # NYU
        raw_data = np.array(mat_data['AAL'])[0]
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab'][0])

    elif args.dataset == 2:  # UM116
        raw_data = np.array(mat_data['AAL'])[0]
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab'][0])

    elif args.dataset == 3:  # ADNI
        raw_data = np.array(mat_data['timeseries'])
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        data = np.transpose(data, (0, 2, 1))
        labels = np.array(mat_data['label'][0])

        # 打印原始标签分布
        print("\nADNI原始标签分布:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"标签 {label}: {count} 个样本 ({count / len(labels) * 100:.2f}%)")

    elif args.dataset == 4:  # NYU200
        raw_data = np.array(mat_data['AAL2'])
        data = raw_data[0]
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab']).squeeze()
        labels[labels == -1] = 0

    elif args.dataset == args.um200:
        raw_data = np.array(mat_data['AAL'])
        data = raw_data[0]
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab']).squeeze()
        labels[labels == -1] = 0

    elif args.dataset == args.SITE1:
        raw_data = np.array(mat_data['AAL'])[0]
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab'][0])

    elif args.dataset == args.SITE6:
        raw_data = np.array(mat_data['AAL'])[0]
        for item in raw_data:
            data.append(item)
        data = np.array(np.stack(data, axis=0), np.float64)
        labels = np.array(mat_data['lab'][0])

    # 数据集分割
    train_data, test_data, train_label, test_label = train_test_split(
        data, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )

    if train:
        return train_data, train_label
    else:
        return {tensor(test_data, dtype=float32), tensor(test_label)}


class Load_Data(Dataset):
    def __init__(self, k_fold=None):
        self.data_dict = {}
        self.data, self.label = dataread()

        # 构建数据字典
        for id in range(self.data.shape[0]):
            self.data_dict[id] = self.data[id, :, :]

        self.full_subject_list = list(self.data_dict.keys())

        # K折交叉验证设置
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=args.seed)
        self.k = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        """设置交叉验证折"""
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.label))[fold]

        if train:
            shuffle(train_idx)
        if not train:
            shuffle(test_idx)

        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        X = self.data_dict[subject]
        y = self.label[subject]

        # 数据标准化
        if args.normalize_data:
            X = (X - X.mean()) / (X.std() + 1e-8)

        return {'id': subject, 'X': tensor(X, dtype=float32), 'y': y}


class BrainConnectivityDataset(Dataset):
    def __init__(self, features, labels):
        """初始化数据集
        Args:
            features: 特征张量 [num_subjects, num_rois, timepoints]
            labels: 标签张量 [num_subjects]
        """
        # 确保输入是张量
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features)
        if not isinstance(labels, torch.Tensor):
            labels = torch.LongTensor(labels)

        self.features = features
        self.labels = labels

        # 打印数据集信息
        print("\n数据集信息:")
        print(f"特征形状: {self.features.shape}")
        print(f"标签形状: {self.labels.shape}")
        print(f"类别分布: {torch.bincount(self.labels).tolist()}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)

    def __getitem__(self, idx):
        """获取单个样本
        Args:
            idx: 样本索引
        Returns:
            (feature, label): 特征和标签对
        """
        features = self.features[idx]
        labels = self.labels[idx]

        # 确保特征标准化
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features, labels

    def get_input_shape(self):
        """获取输入特征的形状"""
        return self.features.shape[1:]  # [num_rois, timepoints]

    def get_num_classes(self):
        """获取类别数量"""
        return len(torch.unique(self.labels))

    def get_class_weights(self):
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = torch.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights



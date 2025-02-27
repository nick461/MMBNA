import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import torch.nn.functional as F


class MultiViewGenerator:
    def __init__(self, node_mask_ratio=0.2, edge_mask_ratio=0.2, feature_mask_ratio=0.4):
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio
        self.device = None  # 添加设备属性
        self.training = True  # 添加训练模式标志

        # 添加随机种子
        self.rng = np.random.RandomState(42)

    def to(self, device):
        """将生成器移动到指定设备"""
        self.device = device
        return self

    def calculate_pc_gpu(self, x, y, device):
        """GPU版本的皮尔逊相关系数计算"""
        x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x.to(device)
        y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y.to(device)

        x_mean = x.mean()
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
        return numerator / (denominator + 1e-8)

    def calculate_sr_improved(self, X, lambda_val=1.0):
        """按照公式(7)实现稀疏表示"""
        n_features = X.shape[0]
        S = np.zeros((n_features, n_features))

        def objective_function(s, x_i, X_others):
            reconstruction = X_others @ s
            reconstruction_error = np.sum((x_i - reconstruction) ** 2)
            l1_penalty = lambda_val * np.sum(np.abs(s))
            return reconstruction_error + l1_penalty

        for i in range(n_features):
            x_i = X[i]
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            X_others = X[mask].T

            s0 = np.zeros(n_features - 1)
            result = minimize(
                objective_function,
                s0,
                args=(x_i, X_others),
                method='L-BFGS-B',
                options={'maxiter': 100}
            )

            s = np.zeros(n_features)
            s[mask] = result.x
            S[i] = s

        # 确保矩阵对称
        S = (S + S.T) / 2
        return S

    def calculate_mi_improved(self, x, y, kernel_width=0.1):
        """按照公式(8)实现互信息，添加预处理和错误处理"""
        try:
            # 数据预处理
            x = np.array(x).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)

            # 添加小的随机噪声以避免奇异性
            x = x + np.random.normal(0, 1e-10, x.shape)
            y = y + np.random.normal(0, 1e-10, y.shape)

            # 标准化
            x = (x - np.mean(x)) / (np.std(x) + 1e-10)
            y = (y - np.mean(y)) / (np.std(y) + 1e-10)

            # 连接数据
            xy = np.concatenate([x, y], axis=1)

            try:
                # 尝试使用gaussian_kde
                kde_joint = gaussian_kde(xy.T, bw_method=kernel_width)
                kde_x = gaussian_kde(x.T, bw_method=kernel_width)
                kde_y = gaussian_kde(y.T, bw_method=kernel_width)

                # 计算互信息
                mi = 0
                n_samples = len(x)

                for i in range(n_samples):
                    p_joint = kde_joint.evaluate(xy[i].reshape(-1, 1))
                    p_x = kde_x.evaluate(x[i].reshape(-1, 1))
                    p_y = kde_y.evaluate(y[i].reshape(-1, 1))

                    if p_joint > 0 and p_x > 0 and p_y > 0:
                        mi += np.log(p_joint / (p_x * p_y))

                return mi / n_samples

            except np.linalg.LinAlgError:
                # 如果gaussian_kde失败，使用替代方法
                # 这里使用简化的互信息估计
                corr = np.corrcoef(x.ravel(), y.ravel())[0, 1]
                # 使用相关系数的绝对值作为互信息的近似
                return np.abs(corr)

        except Exception as e:
            print(f"Warning: MI calculation failed with error {str(e)}")
            # 返回一个默认值
            return 0.0

    def calculate_correlation_matrices(self, x, device):
        """计算三种相关性矩阵 - 批量计算版本"""
        batch_size, n_rois, n_timepoints = x.shape

        # 1. 皮尔逊相关系数 - 直接在GPU上批量计算
        x_normalized = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
        pc_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2)) / n_timepoints

        # 2. 互信息矩阵 - 使用批量计算
        # 使用皮尔逊相关系数的绝对值作为互信息的近似
        mi_matrix = torch.abs(pc_matrix)  # 直接使用pc_matrix的绝对值

        # 3. 稀疏表示矩阵 - 使用批量计算
        # 使用相关性矩阵的软阈值作为稀疏表示
        threshold = 0.1
        sr_matrix = torch.sign(pc_matrix) * F.relu(torch.abs(pc_matrix) - threshold)

        return pc_matrix, mi_matrix, sr_matrix

    def calculate_mi_gpu(self, x, y):
        """GPU版本的互信息计算"""
        # 将数据重塑并标准化
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        # 使用简化的互信息估计 - 直接在GPU上计算
        xy = torch.stack([x, y], dim=0)
        corr = torch.corrcoef(xy)[0, 1]
        return torch.abs(corr)  # 使用相关系数的绝对值作为互信息的近似

    def calculate_sr_gpu(self, x):
        """GPU版本的稀疏表示计算"""
        batch_size, n_rois, n_timepoints = x.shape
        sr_matrix = torch.zeros((batch_size, n_rois, n_rois), device=x.device)

        # 对每个批次单独处理
        for b in range(batch_size):
            # 标准化数据
            x_norm = (x[b] - x[b].mean(dim=1, keepdim=True)) / (x[b].std(dim=1, keepdim=True) + 1e-8)

            # 计算相关性矩阵
            corr_matrix = torch.mm(x_norm, x_norm.t()) / n_timepoints

            # 使用相关性矩阵的软阈值作为稀疏表示
            threshold = 0.1
            sr_matrix[b] = torch.sign(corr_matrix) * torch.relu(torch.abs(corr_matrix) - threshold)

        return sr_matrix

    def train(self):
        """设置为训练模式"""
        self.training = True
        return self

    def eval(self):
        """设置为评估模式"""
        self.training = False
        return self

    def generate_all_views(self, x, ablation_single_view=False):
        """生成所有视图
        Args:
            x: [batch_size, 116, 137]  # 直接使用原始特征形状
        """
        batch_size = x.size(0)
        n_rois = x.size(1)  # 116
        device = x.device

        # 计算相关性矩阵
        x_normalized = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
        pc_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # [batch_size, 116, 116]

        if ablation_single_view:
            views = []
            for _ in range(3):
                views.extend([
                    pc_matrix,  # 节点视图 [batch_size, 116, 116]
                    pc_matrix,  # 边视图 [batch_size, 116, 116]
                    x  # 特征视图 [batch_size, 116, 137]
                ])
            return views

        # 多视图模式：计算额外的相关性矩阵
        # 2. 互信息矩阵
        mi_matrix = torch.abs(pc_matrix)  # 使用相关系数的绝对值作为互信息的近似

        # 3. 稀疏表示矩阵
        threshold = 0.1
        sr_matrix = torch.sign(pc_matrix) * F.relu(torch.abs(pc_matrix) - threshold)

        # 4. 批量生成所有掩码 - 不使用固定种子
        node_masks = torch.rand(3, batch_size, n_rois, device=device) > self.node_mask_ratio
        edge_masks = torch.rand(3, batch_size, n_rois, n_rois, device=device) > self.edge_mask_ratio

        # 转换为float
        node_masks = node_masks.float()
        edge_masks = edge_masks.float()

        # 确保边掩码对称性
        edge_masks = (edge_masks + edge_masks.transpose(2, 3)) / 2

        views = []
        correlation_matrices = [pc_matrix, mi_matrix, sr_matrix]

        # 5. 批量应用掩码
        for i, corr_matrix in enumerate(correlation_matrices):
            # 节点掩码视图
            node_mask_matrix = node_masks[i].unsqueeze(-1) * node_masks[i].unsqueeze(1)
            views.append(corr_matrix * node_mask_matrix)

            # 边掩码视图
            views.append(corr_matrix * edge_masks[i])

            # 特征掩码视图
            views.append(self.generate_feature_masked_view(x))

        return views

    def apply_node_mask(self, x, correlation_matrix):
        """对相关性矩阵应用节点掩码
        Args:
            x: [batch_size, n_rois, timepoints] (实际上不需要使用)
            correlation_matrix: [batch_size, n_rois, n_rois]
        Returns:
            masked_correlation: [batch_size, n_rois, n_rois]
        """
        batch_size, n_rois, _ = x.shape
        device = x.device

        # 生成节点掩码
        node_mask = torch.rand(batch_size, n_rois, device=device) > self.node_mask_ratio
        node_mask = node_mask.float()

        # 将节点掩码应用到相关性矩阵
        node_mask_matrix = node_mask.unsqueeze(-1) * node_mask.unsqueeze(1)
        masked_correlation = correlation_matrix * node_mask_matrix

        return masked_correlation

    def apply_edge_mask(self, x, correlation_matrix):
        """对相关性矩阵应用边掩码
        Args:
            x: [batch_size, n_rois, timepoints] (实际上不需要使用)
            correlation_matrix: [batch_size, n_rois, n_rois]
        Returns:
            masked_correlation: [batch_size, n_rois, n_rois]
        """
        batch_size, n_rois, _ = x.shape
        device = x.device

        # 生成边掩码
        edge_mask = torch.rand(batch_size, n_rois, n_rois, device=device) > self.edge_mask_ratio
        edge_mask = edge_mask.float()

        # 确保对称性
        edge_mask = (edge_mask + edge_mask.transpose(1, 2)) / 2

        # 应用边掩码到相关性矩阵
        masked_correlation = correlation_matrix * edge_mask

        return masked_correlation

    def generate_feature_masked_view(self, x):
        """生成特征掩码视图"""
        batch_size, n_rois, n_timepoints = x.shape
        device = x.device

        # 生成特征掩码
        feature_mask = torch.rand(batch_size, 1, n_timepoints, device=device) > self.feature_mask_ratio
        feature_mask = feature_mask.float()
        feature_mask = feature_mask.expand(-1, n_rois, -1)

        # 应用掩码
        masked_x = x * feature_mask
        return masked_x


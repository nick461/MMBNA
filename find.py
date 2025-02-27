import numpy as np

def find_top_correlations(correlation_matrix, n=10):
    """
    从相关系数矩阵中找出最强的n个独特相关性，返回一个对称矩阵
    Args:
        correlation_matrix: 90x90的相关系数矩阵
        n: 需要保留的最强独特连接数量
    Returns:
        90×90的稀疏对称相关系数矩阵，保留主对角线
    """
    # 获取上三角矩阵（不包括对角线）
    triu_mask = np.triu(np.ones_like(correlation_matrix), k=1)
    upper_triangle = correlation_matrix * triu_mask
    
    # 将上三角矩阵转换为一维数组（忽略0值）
    correlations = upper_triangle[upper_triangle != 0]
    
    # 获取非零元素的位置
    rows, cols = np.nonzero(upper_triangle)
    
    # 按相关系数的绝对值排序
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    
    # 创建新的矩阵，初始化对角线为1
    result_matrix = np.eye(correlation_matrix.shape[0])
    
    # 保留最强的n个独特连接（同时设置对称位置）
    for i in range(n):
        idx = sorted_indices[i]
        row, col = rows[idx], cols[idx]
        value = correlations[idx]
        result_matrix[row, col] = value
        result_matrix[col, row] = value  # 设置对称位置
    
    return result_matrix

def main():
    try:
        # 读取相关系数矩阵
        correlation_matrix = np.loadtxt('pearson_correlation_results.txt')
        print(f"成功读取相关系数矩阵，形状: {correlation_matrix.shape}")
        
        # 找出相关性最强的10个独特连接
        result_matrix = find_top_correlations(correlation_matrix, n=10)
        
        # 保存结果到文件
        np.savetxt('max_10.txt', result_matrix, fmt='%8.4f')
        
        # 只显示上三角的非零元素（排除对角线）
        triu_mask = np.triu(np.ones_like(result_matrix), k=1)
        non_zero = np.nonzero(result_matrix * triu_mask)
        print("\n最强的10个独特连接:")
        for i in range(len(non_zero[0])):
            row = non_zero[0][i]
            col = non_zero[1][i]
            value = result_matrix[row, col]
            print(f"ROI {row:3d} <-> ROI {col:3d}: {value:8.4f}")
        
        print(f"\n结果矩阵已保存到 max_10.txt (形状: {result_matrix.shape})")
            
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

if __name__ == '__main__':
    main()

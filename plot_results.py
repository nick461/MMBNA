import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_results_file(file_path):
    """解析结果文件中的指标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配所有需要的指标
        metrics = {}
        patterns = {
            'accuracy': r"Accuracy:\s*([\d.]+)%",
            'precision': r"Precision:\s*([\d.]+)%",
            'f1': r"F1-score:\s*([\d.]+)%"
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[metric_name] = float(match.group(1))
            else:
                print(f"无法在文件 {file_path} 中找到 {metric_name} 数据")
                metrics[metric_name] = None
                
        return metrics
            
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    
    return None

def plot_mask_ratio_impact(results_dict):
    """绘制不同掩码率对模型性能的影响"""
    plt.figure(figsize=(16, 6))
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 准备数据
    mask_ratios = list(results_dict.keys())
    accuracies = [results[0] for results in results_dict.values()]
    precisions = [results[1] for results in results_dict.values()]
    f1_scores = [results[2] for results in results_dict.values()]
    
    # 计算合适的y轴范围
    y_min = 80  # 设置y轴最小值为80
    y_max = 95  # 设置y轴最大值为95
    
    # 设置背景网格
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 绘制三条曲线
    plt.plot(mask_ratios, accuracies, 
            marker='o', 
            markersize=10,
            linewidth=3,
            color='#2878B5',
            label='Accuracy',
            linestyle='-',
            zorder=3)
            
    plt.plot(mask_ratios, precisions, 
            marker='s', 
            markersize=10,
            linewidth=3,
            color='#E68310',
            label='Precision',
            linestyle='--',
            zorder=3)
            
    plt.plot(mask_ratios, f1_scores, 
            marker='^', 
            markersize=10,
            linewidth=3,
            color='#207F4C',
            label='F1-score',
            linestyle=':',
            zorder=3)
    
    # 添加0.05处的红色虚线
    optimal_x = 0.05
    plt.axvline(x=optimal_x, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # 设置坐标轴
    plt.xlabel('Mask Ratio', fontsize=20, labelpad=10, fontweight='bold')
    plt.ylabel('Metrics (%)', fontsize=20, labelpad=10, fontweight='bold')
    
    # 设置y轴范围
    plt.ylim(y_min, y_max)
    
    # 设置x轴刻度和字体
    plt.xticks(mask_ratios, [f'{ratio:.2f}' for ratio in mask_ratios], fontsize=16)
    plt.yticks(np.arange(y_min, y_max + 1, 5), fontsize=16)
    
    # 修改图例样式
    plt.legend(fontsize=14,  # 减小字体大小
              loc='upper right', 
              framealpha=0.9,
              edgecolor='gray',
              bbox_to_anchor=(1.0, 1.0),  # 调整位置
              borderaxespad=0.1,  # 减小边距
              handlelength=1.5,   # 减小图例线的长度
              handletextpad=0.5,  # 减小图例标记和文本之间的间距
              borderpad=0.2)      # 减小边框内边距
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join('mask_result', 'mask_ratio_impact.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n已生成掩码率影响图: {save_path}")

def main():
    result_dir = 'mask_result'
    results_dict = {}
    
    # 确保目录存在
    if not os.path.exists(result_dir):
        print(f"目录 {result_dir} 不存在")
        return
    
    print(f"\n读取 {result_dir} 目录中的结果文件...")
    
    # 收集所有掩码率的结果
    for file_name in os.listdir(result_dir):
        if not file_name.endswith('.txt'):
            continue
            
        try:
            # 从文件名中提取掩码率
            mask_rate = float(file_name.split('.txt')[0])
            file_path = os.path.join(result_dir, file_name)
            
            # 读取文件内容并解析所有指标
            metrics = parse_results_file(file_path)
            if metrics:
                results_dict[mask_rate] = (
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['f1']
                )
                print(f"掩码率 {mask_rate:.1f}: Acc={metrics['accuracy']:.2f}%, "
                      f"Pre={metrics['precision']:.2f}%, F1={metrics['f1']:.2f}%")
                
        except Exception as e:
            if not file_name.startswith('.'):  # 忽略隐藏文件
                print(f"跳过文件 {file_name}: {str(e)}")
    
    if results_dict:
        # 按掩码率排序
        sorted_results = dict(sorted(results_dict.items()))
        
        # 绘制图表
        plot_mask_ratio_impact(sorted_results)
    else:
        print("未找到有效的结果文件")

if __name__ == '__main__':
    main() 
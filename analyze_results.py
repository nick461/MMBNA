import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results_dir):
    """绘制实验结果比较图"""
    # 读取所有实验结果
    all_results = {}
    for exp_name in os.listdir(results_dir):
        result_path = os.path.join(results_dir, exp_name, 'results.json')
        if os.path.exists(result_path):
            with open(result_path) as f:
                all_results[exp_name] = json.load(f)
    
    # 绘制损失比较图
    plt.figure(figsize=(15, 10))
    
    # 总损失比较
    plt.subplot(2, 2, 1)
    exp_names = list(all_results.keys())
    val_losses = [r['best_val_loss'] for r in all_results.values()]
    sns.barplot(x=exp_names, y=val_losses)
    plt.title('Best Validation Loss Comparison')
    plt.xticks(rotation=45)
    
    # 损失分解比较
    plt.subplot(2, 2, 2)
    loss_types = ['recon_loss', 'kl_loss', 'consistency_loss']
    for exp_name, result in all_results.items():
        losses = [result['val_losses'][lt] for lt in loss_types]
        plt.plot(loss_types, losses, marker='o', label=exp_name)
    plt.legend()
    plt.title('Loss Components Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison.png'))
    plt.close()

if __name__ == '__main__':
    plot_results('./checkpoints') 
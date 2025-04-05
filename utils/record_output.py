import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_record, save_path=None):
    """
    绘制损失曲线
    Args:
        loss_record: 损失值列表
        save_path: 图片保存路径（如不提供则只显示不保存）
    """
    if not loss_record or len(loss_record) == 0:
        print("Warning: loss_record is empty!")
        return
    
    plt.figure(figsize=(10, 5))
    
    # 绘制原始损失点
    plt.scatter(
        range(len(loss_record)), 
        loss_record, 
        s=3, 
        alpha=0.3, 
        label='Raw Loss'
    )
    
    # 计算滑动平均（窗口大小=50）
    window_size = min(50, len(loss_record))  # 确保窗口不超过数据长度
    if window_size > 1:  # 只有数据足够时才计算移动平均
        moving_avg = np.convolve(
            loss_record, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        
        # 绘制移动平均线
        # x轴从(window_size//2)开始，到(len(loss_record)-window_size//2)
        x_vals = range(window_size//2, len(loss_record)-window_size//2 + 1)
        plt.plot(
            x_vals, 
            moving_avg, 
            color='red',
            linewidth=2,
            label=f'Moving Avg (window={window_size})'
        )
    
    plt.title('Training Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    plt.show()

def plot_tard(data, title="Tard Curve", save_path=None):
    """
    专门针对短数据列表的绘图函数
    Args:
        data: 输入数据列表 (示例: [409.0, 324.0, 757.0, 352.0, 893.0])
        title: 图表标题
        save_path: 图片保存路径（可选）
    """
    if not data:
        print("错误：数据为空！")
        return
    
    plt.figure(figsize=(10, 5))
    
    # 创建x轴坐标（从1开始而不是0，更直观）
    x_values = range(1, len(data)+1)
    
    # 绘制折线图（带标记点）
    plt.plot(x_values, data, 
             marker='o',         # 圆形标记点
             markersize=8,       # 标记大小
             linewidth=2,        # 线宽
             color='royalblue',  # 线条颜色
             label='Raw Data')
    
    # 添加数据标签
    for x, y in zip(x_values, data):
        plt.text(x, y, f'{y:.1f}', 
                 ha='center',     # 水平居中
                 va='bottom',    # 垂直位于点上
                 fontsize=10)
    
    # 图表装饰
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.xticks(x_values)  # 确保x轴显示所有刻度
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # 自动调整y轴范围（留出顶部空间放标签）
    y_min, y_max = min(data), max(data)
    plt.ylim(y_min*0.9, y_max*1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.show()
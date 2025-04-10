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
    # plt.show()


def plot_tard(loss_record, save_path=None):
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
    
    # 计算滑动平均（窗口大小=20）
    window_size = min(20, len(loss_record))  # 确保窗口不超过数据长度
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
    plt.ylabel('Tard Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"tard plot saved to {save_path}")
    # plt.show()
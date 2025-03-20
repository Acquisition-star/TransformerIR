import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 生成示例大数据集（100次迭代）
data = {
    'iter': range(100),
    'lr': [2e-4 * (0.95) ** i for i in range(100)],
    'loss': [0.99 * (0.98) ** i + 0.02 * i for i in range(100)]
}
df = pd.DataFrame(data)

# 创建画布和垂直排列的双子图（更适应长序列）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 通用样式设置
plot_config = {
    "linewidth": 1.5,
    "alpha": 0.8,
    "marker": "",  # 移除数据点标记
    "markersize": 0
}

# 上：学习率曲线（对数坐标）
sns.lineplot(
    data=df, x='iter', y='lr',
    ax=ax1, color='royalblue',
    **plot_config
)
ax1.set_yscale('log')  # 对数坐标转换
ax1.grid(True, which='both', linestyle=':', alpha=0.5)
ax1.set_ylabel('Learning Rate', labelpad=10)

# 下：损失曲线
sns.lineplot(
    data=df, x='iter', y='loss',
    ax=ax2, color='crimson',
    **plot_config
)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlabel('Iteration', labelpad=10)
ax2.set_ylabel('Loss', labelpad=10)

# X轴优化
for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(MaxNLocator(10))  # 自动间隔
    ax.tick_params(axis='x', rotation=45)  # 旋转标签

# 紧凑布局
plt.tight_layout(h_pad=3.0)  # 控制子图垂直间距
plt.subplots_adjust(top=0.92)  # 顶部留出标题空间
plt.suptitle('Training Process Monitoring', y=0.97)

# plt.title('Learning Rate and Loss During Training')
plt.savefig('training_plot.png', dpi=300, bbox_inches='tight')  # 保存图像

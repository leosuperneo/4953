# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import math


class SignalPropagationModel(nn.Module):
    """信号传播模型 - 预测给定AP位置的信号强度分布"""
    
    def __init__(self, grid_size: int = 50):
        super(SignalPropagationModel, self).__init__()
        self.grid_size = grid_size
        
        # 深度神经网络来学习复杂的信号传播特性
        self.network = nn.Sequential(
            nn.Linear(4, 128),  # 输入: [x_ap, y_ap, x_point, y_point]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 1),  # 输出: 信号强度 (dBm)
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
        
    def forward(self, ap_positions: torch.Tensor, test_points: torch.Tensor) -> torch.Tensor:
        """
        计算给定AP位置在测试点的信号强度
        
        Args:
            ap_positions: [num_aps, 2] - AP的坐标
            test_points: [num_points, 2] - 测试点坐标
            
        Returns:
            signal_strength: [num_points] - 每个测试点的信号强度
        """
        num_aps = ap_positions.shape[0]
        num_points = test_points.shape[0]
        
        # 扩展维度以计算所有AP到所有测试点的信号
        ap_expanded = ap_positions.unsqueeze(1).expand(num_aps, num_points, 2)
        points_expanded = test_points.unsqueeze(0).expand(num_aps, num_points, 2)
        
        # 组合输入: [num_aps, num_points, 4]
        inputs = torch.cat([ap_expanded, points_expanded], dim=-1)
        inputs = inputs.reshape(-1, 4)
        
        # 预测信号强度
        signals = self.network(inputs).reshape(num_aps, num_points)
        
        # 取最强信号 (多个AP时)
        max_signal, _ = torch.max(signals, dim=0)
        
        return max_signal


class APPositionOptimizer(nn.Module):
    """AP位置优化器 - 学习最优的AP部署位置"""
    
    def __init__(self, num_aps: int, grid_size: Tuple[int, int] = (50, 50)):
        super(APPositionOptimizer, self).__init__()
        self.num_aps = num_aps
        self.grid_size = grid_size
        
        # 使用可学习参数表示AP位置
        # 初始化为网格上的随机位置
        initial_positions = torch.rand(num_aps, 2)
        initial_positions[:, 0] *= grid_size[0]
        initial_positions[:, 1] *= grid_size[1]
        
        self.ap_positions = nn.Parameter(initial_positions)
        
    def forward(self) -> torch.Tensor:
        """返回当前的AP位置，并确保在有效范围内"""
        # 使用sigmoid将位置限制在网格范围内
        positions = torch.stack([
            torch.sigmoid(self.ap_positions[:, 0]) * self.grid_size[0],
            torch.sigmoid(self.ap_positions[:, 1]) * self.grid_size[1]
        ], dim=1)
        return positions


class MineEnvironment:
    """矿井环境模拟 - 生成训练数据"""
    
    def __init__(self, grid_size: Tuple[int, int] = (50, 50), 
                 obstacles: List[Tuple[int, int, int, int]] = None):
        """
        Args:
            grid_size: 矿井网格大小 (宽, 高)
            obstacles: 障碍物列表 [(x1, y1, x2, y2), ...] 表示矩形障碍物
        """
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        
        # 物理信号传播参数
        self.path_loss_exponent = 3.0  # 路径损耗指数
        self.reference_distance = 1.0   # 参考距离
        self.reference_power = -30.0    # 参考功率 (dBm)
        self.wall_attenuation = 15.0    # 墙体衰减 (dB)
        
    def calculate_signal_strength(self, ap_pos: np.ndarray, 
                                 point_pos: np.ndarray) -> float:
        """
        使用物理模型计算信号强度
        基于对数距离路径损耗模型
        """
        # 计算欧几里得距离
        distance = np.linalg.norm(ap_pos - point_pos)
        if distance < 0.1:
            distance = 0.1
            
        # 基础路径损耗
        path_loss = self.reference_power - 10 * self.path_loss_exponent * \
                    np.log10(distance / self.reference_distance)
        
        # 检查是否穿过障碍物
        num_walls = self._count_wall_intersections(ap_pos, point_pos)
        wall_loss = num_walls * self.wall_attenuation
        
        # 总信号强度
        signal_strength = path_loss - wall_loss
        
        # 归一化到 [0, 1]
        normalized = (signal_strength + 100) / 100  # 假设信号范围 [-100, 0] dBm
        return np.clip(normalized, 0, 1)
    
    def _count_wall_intersections(self, pos1: np.ndarray, 
                                  pos2: np.ndarray) -> int:
        """计算线段穿过的障碍物数量"""
        count = 0
        for obstacle in self.obstacles:
            x1, y1, x2, y2 = obstacle
            # 简化：检查线段是否与矩形相交
            if self._line_intersects_rect(pos1, pos2, (x1, y1, x2, y2)):
                count += 1
        return count
    
    def _line_intersects_rect(self, p1: np.ndarray, p2: np.ndarray,
                             rect: Tuple[int, int, int, int]) -> bool:
        """检查线段是否与矩形相交"""
        x1, y1, x2, y2 = rect
        # 简化的相交检测
        return not (max(p1[0], p2[0]) < x1 or min(p1[0], p2[0]) > x2 or
                   max(p1[1], p2[1]) < y1 or min(p1[1], p2[1]) > y2)
    
    def generate_training_data(self, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练数据"""
        ap_positions = []
        test_points = []
        signal_strengths = []
        
        for _ in range(num_samples):
            # 随机AP位置
            ap_pos = np.array([
                np.random.uniform(0, self.grid_size[0]),
                np.random.uniform(0, self.grid_size[1])
            ])
            
            # 随机测试点
            test_pos = np.array([
                np.random.uniform(0, self.grid_size[0]),
                np.random.uniform(0, self.grid_size[1])
            ])
            
            # 计算信号强度
            signal = self.calculate_signal_strength(ap_pos, test_pos)
            
            ap_positions.append(ap_pos)
            test_points.append(test_pos)
            signal_strengths.append(signal)
        
        # 组合输入特征
        inputs = np.concatenate([
            np.array(ap_positions),
            np.array(test_points)
        ], axis=1)
        
        return torch.FloatTensor(inputs), torch.FloatTensor(signal_strengths)


def train_signal_model(model: SignalPropagationModel, 
                       train_data: Tuple[torch.Tensor, torch.Tensor],
                       epochs: int = 100,
                       batch_size: int = 64,
                       lr: float = 0.001) -> List[float]:
    """训练信号传播模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    inputs, targets = train_data
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    print("开始训练信号传播模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch训练
        indices = torch.randperm(len(inputs))
        for i in range(0, len(inputs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            
            optimizer.zero_grad()
            outputs = model.network(batch_inputs).squeeze()
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(inputs) / batch_size)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return losses


def optimize_ap_positions(signal_model: SignalPropagationModel,
                         num_aps: int,
                         grid_size: Tuple[int, int],
                         num_test_points: int = 500,
                         epochs: int = 200,
                         lr: float = 0.1) -> Tuple[torch.Tensor, List[float]]:
    """优化AP位置以最大化覆盖率"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal_model = signal_model.to(device)
    signal_model.eval()
    
    # 创建位置优化器
    position_optimizer = APPositionOptimizer(num_aps, grid_size).to(device)
    optimizer = optim.Adam(position_optimizer.parameters(), lr=lr)
    
    # 生成测试点网格
    x = torch.linspace(0, grid_size[0], int(np.sqrt(num_test_points)))
    y = torch.linspace(0, grid_size[1], int(np.sqrt(num_test_points)))
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    test_points = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    
    coverage_history = []
    
    print(f"\n开始优化 {num_aps} 个AP的位置...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 获取当前AP位置
        ap_positions = position_optimizer()
        
        # 计算每个测试点的信号强度
        signal_strengths = signal_model(ap_positions, test_points)
        
        # 目标1: 最大化平均信号强度
        avg_signal = signal_strengths.mean()
        
        # 目标2: 最大化覆盖率 (信号 > 阈值的点数)
        threshold = 0.3
        coverage = (signal_strengths > threshold).float().mean()
        
        # 目标3: 最小化信号方差 (使信号分布更均匀)
        signal_variance = signal_strengths.var()
        
        # 目标4: AP之间保持适当距离 (避免过于集中)
        if num_aps > 1:
            distances = torch.cdist(ap_positions, ap_positions)
            # 排除对角线(自己到自己的距离)
            mask = ~torch.eye(num_aps, dtype=torch.bool, device=device)
            min_distances = distances[mask].view(num_aps, num_aps-1).min(dim=1)[0]
            spacing_penalty = torch.exp(-min_distances / 10).mean()
        else:
            spacing_penalty = 0
        
        # 组合损失函数 (负号因为我们要最大化)
        loss = -(2.0 * avg_signal + 3.0 * coverage - 0.5 * signal_variance - 0.5 * spacing_penalty)
        
        loss.backward()
        optimizer.step()
        
        coverage_history.append(coverage.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Coverage: {coverage.item():.4f}, "
                  f"Avg Signal: {avg_signal.item():.4f}, "
                  f"Loss: {loss.item():.4f}")
    
    final_positions = position_optimizer().detach()
    return final_positions, coverage_history


def visualize_results(ap_positions: torch.Tensor,
                     signal_model: SignalPropagationModel,
                     grid_size: Tuple[int, int],
                     obstacles: List[Tuple[int, int, int, int]] = None,
                     save_path: str = None):
    """可视化AP部署和信号覆盖"""
    
    device = ap_positions.device
    
    # 创建密集的测试点网格
    resolution = 100
    x = torch.linspace(0, grid_size[0], resolution)
    y = torch.linspace(0, grid_size[1], resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    test_points = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    
    # 计算信号强度分布
    with torch.no_grad():
        signal_strengths = signal_model(ap_positions, test_points)
    
    signal_map = signal_strengths.cpu().numpy().reshape(resolution, resolution)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 信号强度热力图
    im1 = ax1.imshow(signal_map.T, origin='lower', extent=[0, grid_size[0], 0, grid_size[1]],
                     cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title('Heat map of signal strength distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    
    # 绘制AP位置
    ap_pos_cpu = ap_positions.cpu().numpy()
    ax1.scatter(ap_pos_cpu[:, 0], ap_pos_cpu[:, 1], 
               c='blue', s=300, marker='^', edgecolors='white', linewidths=2,
               label='AP location', zorder=5)
    
    # 标注AP编号
    for i, pos in enumerate(ap_pos_cpu):
        ax1.text(pos[0], pos[1], f'AP{i+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 绘制障碍物
    if obstacles:
        for obs in obstacles:
            x1, y1, x2, y2 = obs
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5)
            ax1.add_patch(rect)
    
    ax1.legend(loc='upper right', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='signal strength (unit)')
    ax1.grid(True, alpha=0.3)
    
    # 覆盖率分析
    threshold_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    coverage_rates = []
    for threshold in threshold_levels:
        coverage = (signal_strengths > threshold).float().mean().item()
        coverage_rates.append(coverage * 100)
    
    ax2.bar(range(len(threshold_levels)), coverage_rates, 
           color='steelblue', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Signal strength threshold', fontsize=12)
    ax2.set_ylabel('fraction of coverage (%)', fontsize=12)
    ax2.set_title('Coverage rates at different thresholds', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(threshold_levels)))
    ax2.set_xticklabels([f'{t:.1f}' for t in threshold_levels])
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(coverage_rates):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()
    
    # 打印详细统计
    print("\n" + "="*50)
    print("AP部署优化结果统计")
    print("="*50)
    print(f"AP数量: {len(ap_positions)}")
    print(f"矿井大小: {grid_size[0]} x {grid_size[1]} 米")
    print("\nAP位置坐标:")
    for i, pos in enumerate(ap_pos_cpu):
        print(f"  AP{i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")
    print(f"\n平均信号强度: {signal_strengths.mean().item():.4f}")
    print(f"信号强度标准差: {signal_strengths.std().item():.4f}")
    print(f"30%阈值覆盖率: {coverage_rates[1]:.2f}%")
    print("="*50)


def main():
    """主函数"""
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 参数配置
    GRID_SIZE = (50, 50)  # 矿井大小 (米)
    NUM_APS = 4           # AP数量
    
    # 定义障碍物 (模拟矿井中的墙壁、设备等)
    OBSTACLES = [
        (15, 15, 20, 35),  # 矩形障碍物 (x1, y1, x2, y2)
        (30, 10, 35, 25),
        (10, 40, 25, 45),
    ]
    
    print("="*50)
    print("矿井AP部署优化系统")
    print("="*50)
    
    # 步骤1: 创建矿井环境并生成训练数据
    print("\n[1/4] 生成训练数据...")
    env = MineEnvironment(GRID_SIZE, OBSTACLES)
    train_inputs, train_targets = env.generate_training_data(num_samples=5000)
    print(f"生成了 {len(train_inputs)} 个训练样本")
    
    # 步骤2: 训练信号传播模型
    print("\n[2/4] 训练信号传播模型...")
    signal_model = SignalPropagationModel()
    train_losses = train_signal_model(
        signal_model, 
        (train_inputs, train_targets),
        epochs=100,
        batch_size=64
    )
    print("信号传播模型训练完成!")
    
    # 步骤3: 优化AP位置
    print("\n[3/4] 优化AP位置...")
    optimal_positions, coverage_history = optimize_ap_positions(
        signal_model,
        num_aps=NUM_APS,
        grid_size=GRID_SIZE,
        num_test_points=625,  # 25x25网格
        epochs=200
    )
    print("AP位置优化完成!")
    
    # 步骤4: 可视化结果
    print("\n[4/4] 生成可视化结果...")
    visualize_results(
        optimal_positions,
        signal_model,
        GRID_SIZE,
        OBSTACLES,
        save_path='/mnt/user-data/outputs/mine_ap_optimization.png'
    )
    
    print("\n优化完成! 🎉")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:53:07 2025

@author: HONGTAO LEO
"""

"""
WiFi HaLow矿道通信路径规划系统
使用PyTorch深度强化学习优化通信路径
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
from typing import List, Tuple, Dict

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MineTopology:
    """矿道拓扑结构"""
    
    def __init__(self):
        # 定义节点坐标 (x, y)
        self.nodes = {
            'Portal': (0, 0),
            'AP1': (5, 0),
            'Jct_a': (5, 0),
            'Xcutg1_up': (5, 4),
            'STA1': (5, 4),
            'Xcutg2_up': (5, -4),
            'Jct_b': (10, 0),
            'Refuge': (10, 0),
            'Xcutg1_mid': (10, 4),
            'AP2': (10, 4),
            'Xcutg2_mid': (10, -4),
            'STA2': (10, -4),
            'Jct_c': (15, 0),
            'AP3': (15, 2),
            'Stope_F': (18.5, 5.5),
            'STA3': (18.5, 5.5),
            'Stope_E': (18.5, 2),
        }
        
        # 定义连接关系和距离
        self.edges = [
            ('Portal', 'AP1', 5.0),
            ('AP1', 'Jct_a', 0.1),
            ('Jct_a', 'Xcutg1_up', 4.0),
            ('Xcutg1_up', 'STA1', 0.1),
            ('Jct_a', 'Xcutg2_up', 4.0),
            ('Jct_a', 'Jct_b', 5.0),
            ('Jct_b', 'Refuge', 0.1),
            ('Jct_b', 'Xcutg1_mid', 4.0),
            ('Xcutg1_mid', 'AP2', 0.1),
            ('Jct_b', 'Xcutg2_mid', 4.0),
            ('Xcutg2_mid', 'STA2', 0.1),
            ('Jct_b', 'Jct_c', 5.0),
            ('Jct_c', 'AP3', 2.0),
            ('AP3', 'Stope_F', 3.5),
            ('Stope_F', 'STA3', 0.1),
            ('AP3', 'Stope_E', 3.5),
        ]
        
        # 构建邻接表
        self.adjacency = defaultdict(list)
        for src, dst, dist in self.edges:
            self.adjacency[src].append((dst, dist))
            self.adjacency[dst].append((src, dist))  # 双向连接
        
        # 节点索引
        self.node_list = sorted(self.nodes.keys())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # AP节点和STA节点
        self.ap_nodes = ['Portal', 'AP1', 'AP2', 'AP3']
        self.sta_nodes = ['STA1', 'STA2', 'STA3']
        
    def get_distance(self, node1: str, node2: str) -> float:
        """计算两节点间的欧氏距离"""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class WiFiHaLowModel:
    """WiFi HaLow信号传播模型"""
    
    def __init__(self):
        # WiFi HaLow (802.11ah) 参数
        self.frequency = 900  # MHz
        self.tx_power = 20    # dBm
        self.sensitivity = -100  # dBm
        self.max_range = 1000  # 米 (理想情况)
        
    def calculate_path_loss(self, distance: float, environment: str = 'tunnel') -> float:
        """
        计算路径损耗 (dB)
        使用Log-distance路径损耗模型
        """
        if distance == 0:
            return 0
        
        # 参考距离和路径损耗
        d0 = 1.0  # 参考距离 (米)
        
        # 不同环境的路径损耗指数
        if environment == 'tunnel':
            n = 2.5  # 矿道环境
            L0 = 40  # 1米处的损耗 (dB)
        else:
            n = 2.0
            L0 = 40
        
        # Log-distance模型
        path_loss = L0 + 10 * n * np.log10(distance / d0)
        
        # 添加阴影衰落 (随机衰减)
        shadow_fading = np.random.normal(0, 4)  # 标准差4dB
        
        return path_loss + shadow_fading
    
    def calculate_rssi(self, distance: float) -> float:
        """计算接收信号强度指示 (RSSI)"""
        path_loss = self.calculate_path_loss(distance)
        rssi = self.tx_power - path_loss
        return rssi
    
    def is_link_valid(self, distance: float) -> bool:
        """判断链路是否有效"""
        rssi = self.calculate_rssi(distance)
        return rssi > self.sensitivity
    
    def calculate_throughput(self, distance: float) -> float:
        """
        根据距离计算吞吐量 (Mbps)
        WiFi HaLow理论最大速率约为40Mbps
        """
        rssi = self.calculate_rssi(distance)
        snr = rssi - (-90)  # 假设噪声为-90dBm
        
        if snr < 10:
            return 0.1
        elif snr < 20:
            return 5.0
        elif snr < 30:
            return 20.0
        else:
            return 40.0


class DQN(nn.Module):
    """深度Q网络用于路径选择"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class PathPlanningAgent:
    """路径规划智能体"""
    
    def __init__(self, topology: MineTopology, wifi_model: WiFiHaLowModel):
        self.topology = topology
        self.wifi_model = wifi_model
        
        # 状态维度：当前位置 + 目标位置 + 已访问节点信息
        self.state_dim = len(topology.node_list) * 3
        # 动作维度：可能的下一跳节点数量
        self.action_dim = len(topology.node_list)
        
        # 创建Q网络和目标网络
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # 经验回放
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        
        # 训练参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_state(self, current_node: str, target_node: str, visited: set) -> torch.Tensor:
        """构建状态表示"""
        state = np.zeros(self.state_dim)
        
        # 当前位置 one-hot
        current_idx = self.topology.node_to_idx[current_node]
        state[current_idx] = 1
        
        # 目标位置 one-hot
        target_idx = self.topology.node_to_idx[target_node]
        state[len(self.topology.node_list) + target_idx] = 1
        
        # 已访问节点
        for node in visited:
            idx = self.topology.node_to_idx[node]
            state[2 * len(self.topology.node_list) + idx] = 1
        
        return torch.FloatTensor(state)
    
    def select_action(self, state: torch.Tensor, current_node: str, valid_actions: List[str]) -> str:
        """选择动作（epsilon-greedy）"""
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # 只考虑有效动作
                valid_mask = torch.full((self.action_dim,), float('-inf'))
                for action in valid_actions:
                    action_idx = self.topology.node_to_idx[action]
                    valid_mask[action_idx] = q_values[action_idx]
                
                action_idx = torch.argmax(valid_mask).item()
                return self.topology.idx_to_node[action_idx]
    
    def calculate_reward(self, current_node: str, next_node: str, target_node: str, 
                        visited: set, path_length: float) -> float:
        """计算奖励"""
        # 如果到达目标，给予大奖励
        if next_node == target_node:
            return 100.0 - path_length * 0.1
        
        # 如果访问过，给予惩罚
        if next_node in visited:
            return -10.0
        
        # 如果是AP节点，给予小奖励（鼓励使用中继）
        if next_node in self.topology.ap_nodes:
            reward = 5.0
        else:
            reward = 1.0
        
        # 根据距离目标的变化给予奖励
        current_dist = self.topology.get_distance(current_node, target_node)
        next_dist = self.topology.get_distance(next_node, target_node)
        
        if next_dist < current_dist:
            reward += 2.0  # 靠近目标
        else:
            reward -= 1.0  # 远离目标
        
        # 根据信号质量给予奖励
        link_dist = self.topology.get_distance(current_node, next_node)
        rssi = self.wifi_model.calculate_rssi(link_dist)
        if rssi > -70:
            reward += 3.0
        elif rssi > -85:
            reward += 1.0
        else:
            reward -= 2.0
        
        return reward
    
    def train_step(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放中采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor([self.topology.node_to_idx[a] for a in actions])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def find_path(self, start_node: str, target_node: str, max_steps: int = 50) -> Tuple[List[str], float]:
        """使用训练好的模型寻找路径"""
        path = [start_node]
        visited = {start_node}
        current_node = start_node
        total_distance = 0
        
        for step in range(max_steps):
            # 获取有效的下一跳
            valid_neighbors = [neighbor for neighbor, _ in self.topology.adjacency[current_node]
                             if neighbor not in visited]
            
            if not valid_neighbors or current_node == target_node:
                break
            
            # 获取状态并选择动作
            state = self.get_state(current_node, target_node, visited)
            
            # 使用贪婪策略（不探索）
            with torch.no_grad():
                q_values = self.q_network(state)
                valid_q = {}
                for neighbor in valid_neighbors:
                    neighbor_idx = self.topology.node_to_idx[neighbor]
                    valid_q[neighbor] = q_values[neighbor_idx].item()
                
                next_node = max(valid_q.items(), key=lambda x: x[1])[0]
            
            # 更新路径
            distance = self.topology.get_distance(current_node, next_node)
            total_distance += distance
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path, total_distance


class PathPlanningTrainer:
    """路径规划训练器"""
    
    def __init__(self, agent: PathPlanningAgent):
        self.agent = agent
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def train_episode(self, start_node: str, target_node: str, max_steps: int = 50) -> float:
        """训练一个回合"""
        current_node = start_node
        visited = {start_node}
        total_reward = 0
        path_length = 0
        
        for step in range(max_steps):
            # 获取当前状态
            state = self.agent.get_state(current_node, target_node, visited)
            
            # 获取有效动作
            valid_actions = [neighbor for neighbor, _ in self.agent.topology.adjacency[current_node]
                           if neighbor not in visited]
            
            if not valid_actions:
                break
            
            # 选择动作
            action = self.agent.select_action(state, current_node, valid_actions)
            
            # 执行动作
            next_node = action
            distance = self.agent.topology.get_distance(current_node, next_node)
            path_length += distance
            
            # 计算奖励
            reward = self.agent.calculate_reward(current_node, next_node, target_node, 
                                                visited, path_length)
            total_reward += reward
            
            # 判断是否结束
            done = (next_node == target_node)
            
            # 获取下一个状态
            visited.add(next_node)
            next_state = self.agent.get_state(next_node, target_node, visited)
            
            # 存储经验
            self.agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # 训练
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            # 更新当前节点
            current_node = next_node
            
            if done:
                break
        
        # 衰减epsilon
        self.agent.epsilon = max(self.agent.epsilon_min, 
                                self.agent.epsilon * self.agent.epsilon_decay)
        
        return total_reward
    
    def train(self, episodes: int = 1000, update_target_every: int = 10):
        """训练多个回合"""
        print("开始训练...")
        
        for episode in range(episodes):
            # 随机选择起点和终点
            sta_node = random.choice(self.agent.topology.sta_nodes)
            target_node = 'Portal'
            
            # 训练一个回合
            total_reward = self.train_episode(sta_node, target_node)
            self.episode_rewards.append(total_reward)
            
            # 更新目标网络
            if episode % update_target_every == 0:
                self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        print("训练完成！")


def visualize_topology(topology: MineTopology, path: List[str] = None):
    """可视化矿道拓扑和路径"""
    plt.figure(figsize=(14, 8))
    
    # 绘制所有边
    for src, dst, _ in topology.edges:
        x1, y1 = topology.nodes[src]
        x2, y2 = topology.nodes[dst]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)
    
    # 绘制节点
    for node, (x, y) in topology.nodes.items():
        if node in topology.ap_nodes:
            plt.scatter(x, y, c='blue', s=200, marker='s', label='AP' if node == 'Portal' else '', zorder=3)
            plt.text(x, y + 0.5, node, fontsize=8, ha='center')
        elif node in topology.sta_nodes:
            plt.scatter(x, y, c='red', s=200, marker='^', label='STA' if node == 'STA1' else '', zorder=3)
            plt.text(x, y + 0.5, node, fontsize=8, ha='center')
        else:
            plt.scatter(x, y, c='gray', s=50, marker='o', zorder=2)
            plt.text(x, y + 0.5, node, fontsize=6, ha='center', alpha=0.7)
    
    # 绘制路径
    if path:
        for i in range(len(path) - 1):
            x1, y1 = topology.nodes[path[i]]
            x2, y2 = topology.nodes[path[i + 1]]
            plt.arrow(x1, y1, x2 - x1, y2 - y1, 
                     head_width=0.3, head_length=0.3, 
                     fc='green', ec='green', linewidth=2, zorder=4)
    
    plt.xlabel('X (米)', fontsize=12)
    plt.ylabel('Y (米)', fontsize=12)
    plt.title('矿道WiFi HaLow通信拓扑', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    
    return plt


def visualize_training_results(trainer: PathPlanningTrainer):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 奖励曲线
    axes[0].plot(trainer.episode_rewards, alpha=0.3)
    # 移动平均
    window = 50
    if len(trainer.episode_rewards) >= window:
        moving_avg = np.convolve(trainer.episode_rewards, 
                                np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(trainer.episode_rewards)), 
                    moving_avg, linewidth=2, label=f'{window}-episode Moving Avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('训练奖励曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    if trainer.losses:
        axes[1].plot(trainer.losses, alpha=0.3)
        if len(trainer.losses) >= window:
            moving_avg = np.convolve(trainer.losses, 
                                    np.ones(window) / window, mode='valid')
            axes[1].plot(range(window - 1, len(trainer.losses)), 
                        moving_avg, linewidth=2, label=f'{window}-step Moving Avg')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('训练损失曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def main():
    """主函数"""
    print("=" * 60)
    print("WiFi HaLow矿道通信路径规划系统")
    print("=" * 60)
    
    # 创建矿道拓扑
    topology = MineTopology()
    print(f"\n矿道节点数量: {len(topology.nodes)}")
    print(f"AP节点: {topology.ap_nodes}")
    print(f"STA节点: {topology.sta_nodes}")
    
    # 创建WiFi HaLow模型
    wifi_model = WiFiHaLowModel()
    print(f"\nWiFi HaLow参数:")
    print(f"  频率: {wifi_model.frequency} MHz")
    print(f"  发射功率: {wifi_model.tx_power} dBm")
    print(f"  接收灵敏度: {wifi_model.sensitivity} dBm")
    
    # 创建路径规划智能体
    agent = PathPlanningAgent(topology, wifi_model)
    print(f"\nDQN网络结构:")
    print(f"  状态维度: {agent.state_dim}")
    print(f"  动作维度: {agent.action_dim}")
    print(agent.q_network)
    
    # 训练
    trainer = PathPlanningTrainer(agent)
    trainer.train(episodes=1000, update_target_every=10)
    
    # 测试路径规划
    print("\n" + "=" * 60)
    print("测试路径规划结果")
    print("=" * 60)
    
    for sta_node in topology.sta_nodes:
        print(f"\n从 {sta_node} 到 Portal 的最优路径:")
        path, distance = agent.find_path(sta_node, 'Portal')
        print(f"  路径: {' -> '.join(path)}")
        print(f"  总距离: {distance:.2f} 米")
        
        # 计算路径的信号质量
        print(f"  链路质量:")
        for i in range(len(path) - 1):
            link_dist = topology.get_distance(path[i], path[i + 1])
            rssi = wifi_model.calculate_rssi(link_dist)
            throughput = wifi_model.calculate_throughput(link_dist)
            print(f"    {path[i]} -> {path[i + 1]}: "
                  f"距离={link_dist:.1f}m, RSSI={rssi:.1f}dBm, "
                  f"吞吐量={throughput:.1f}Mbps")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 可视化拓扑和路径（以STA3为例）
    path, _ = agent.find_path('STA3', 'Portal')
    plt1 = visualize_topology(topology, path)
    plt1.savefig('/mnt/user-data/outputs/mine_topology_with_path.png', dpi=150, bbox_inches='tight')
    print("拓扑图已保存: mine_topology_with_path.png")
    
    # 可视化训练结果
    plt2 = visualize_training_results(trainer)
    plt2.savefig('/mnt/user-data/outputs/training_results.png', dpi=150, bbox_inches='tight')
    print("训练结果已保存: training_results.png")
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), '/mnt/user-data/outputs/path_planning_model.pth')
    print("\n模型已保存: path_planning_model.pth")
    
    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
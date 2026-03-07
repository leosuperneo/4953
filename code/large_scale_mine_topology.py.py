# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:06:21 2025

@author: HONGTAO LEO
"""

"""
大规模WiFi HaLow矿道通信路径规划系统
主矿道：1200米
分支矿道：600米
AP间隔：50米
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
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LargeScaleMineTopology:
    """大规模矿道拓扑结构"""
    
    def __init__(self, 
                 main_tunnel_length: float = 1200.0,  # 主矿道长度(米)
                 branch_tunnel_length: float = 600.0,  # 分支矿道长度(米)
                 ap_spacing: float = 50.0,              # AP间隔(米)
                 num_branches: int = 3):                # 分支数量
        
        self.main_tunnel_length = main_tunnel_length
        self.branch_tunnel_length = branch_tunnel_length
        self.ap_spacing = ap_spacing
        self.num_branches = num_branches
        
        self.nodes = {}
        self.edges = []
        self.adjacency = defaultdict(list)
        
        # 生成拓扑
        self._generate_topology()
        
        # 构建邻接表
        self._build_adjacency()
        
        # 节点索引
        self.node_list = sorted(self.nodes.keys())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        print(f"\n拓扑结构生成完成:")
        print(f"  总节点数: {len(self.nodes)}")
        print(f"  AP节点数: {len(self.ap_nodes)}")
        print(f"  STA节点数: {len(self.sta_nodes)}")
        print(f"  边数量: {len(self.edges)}")
    
    def _generate_topology(self):
        """生成大规模矿道拓扑"""
        
        # 1. 地面入口
        self.nodes['Portal'] = (0, 0)
        
        # 2. 主矿道 - 沿X轴延伸
        num_main_aps = int(self.main_tunnel_length / self.ap_spacing)
        main_ap_positions = []
        
        for i in range(num_main_aps + 1):
            x = i * self.ap_spacing
            node_name = f'Main_AP{i}'
            self.nodes[node_name] = (x, 0)
            main_ap_positions.append(node_name)
            
            # 连接到前一个节点
            if i == 0:
                self.edges.append(('Portal', node_name, self.ap_spacing))
            else:
                prev_node = f'Main_AP{i-1}'
                self.edges.append((prev_node, node_name, self.ap_spacing))
        
        # 3. 分支矿道 - 在主矿道的不同位置分叉
        branch_positions = [
            int(num_main_aps * 0.25),  # 25%位置
            int(num_main_aps * 0.5),   # 50%位置
            int(num_main_aps * 0.75),  # 75%位置
        ]
        
        for branch_idx, main_pos in enumerate(branch_positions[:self.num_branches]):
            junction_node = f'Main_AP{main_pos}'
            junction_x, junction_y = self.nodes[junction_node]
            
            # 创建分支矿道（向上和向下各一条）
            for direction in ['Up', 'Down']:
                num_branch_aps = int(self.branch_tunnel_length / self.ap_spacing)
                y_direction = 1 if direction == 'Up' else -1
                
                for i in range(1, num_branch_aps + 1):
                    y = junction_y + (i * self.ap_spacing * y_direction)
                    node_name = f'Branch{branch_idx+1}_{direction}_AP{i}'
                    self.nodes[node_name] = (junction_x, y)
                    
                    # 连接到前一个节点
                    if i == 1:
                        self.edges.append((junction_node, node_name, self.ap_spacing))
                    else:
                        prev_node = f'Branch{branch_idx+1}_{direction}_AP{i-1}'
                        self.edges.append((prev_node, node_name, self.ap_spacing))
                
                # 在分支末端添加STA（工作面）
                last_ap = f'Branch{branch_idx+1}_{direction}_AP{num_branch_aps}'
                sta_name = f'STA_Branch{branch_idx+1}_{direction}'
                last_x, last_y = self.nodes[last_ap]
                self.nodes[sta_name] = (last_x, last_y + (10 * y_direction))
                self.edges.append((last_ap, sta_name, 10.0))
        
        # 4. 主矿道末端STA
        last_main_ap = f'Main_AP{num_main_aps}'
        main_sta_name = 'STA_MainEnd'
        last_x, _ = self.nodes[last_main_ap]
        self.nodes[main_sta_name] = (last_x + 50, 0)
        self.edges.append((last_main_ap, main_sta_name, 50.0))
        
        # 5. 添加一些中间节点（避难所、交叉口等）
        refuge_positions = [
            int(num_main_aps * 0.33),
            int(num_main_aps * 0.67),
        ]
        
        for i, pos in enumerate(refuge_positions):
            ap_node = f'Main_AP{pos}'
            refuge_name = f'Refuge{i+1}'
            x, y = self.nodes[ap_node]
            self.nodes[refuge_name] = (x, y)
            # 避难所与AP在同一位置，距离很近
            self.edges.append((ap_node, refuge_name, 0.1))
        
        # 确定AP节点和STA节点
        self.ap_nodes = ['Portal'] + [n for n in self.nodes.keys() if 'AP' in n]
        self.sta_nodes = [n for n in self.nodes.keys() if 'STA' in n]
        
    def _build_adjacency(self):
        """构建邻接表"""
        for src, dst, dist in self.edges:
            self.adjacency[src].append((dst, dist))
            self.adjacency[dst].append((src, dist))  # 双向连接
    
    def get_distance(self, node1: str, node2: str) -> float:
        """计算两节点间的欧氏距离"""
        if node1 not in self.nodes or node2 not in self.nodes:
            return float('inf')
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_node_info(self):
        """获取节点信息统计"""
        info = {
            'total_nodes': len(self.nodes),
            'ap_nodes': len(self.ap_nodes),
            'sta_nodes': len(self.sta_nodes),
            'main_tunnel_aps': len([n for n in self.ap_nodes if 'Main_AP' in n]),
            'branch_tunnel_aps': len([n for n in self.ap_nodes if 'Branch' in n]),
            'total_edges': len(self.edges),
        }
        return info


class WiFiHaLowModel:
    """WiFi HaLow信号传播模型"""
    
    def __init__(self):
        # WiFi HaLow (802.11ah) 参数
        self.frequency = 900  # MHz
        self.tx_power = 20    # dBm
        self.sensitivity = -100  # dBm
        self.max_range = 1000  # 米
        
    def calculate_path_loss(self, distance: float, environment: str = 'tunnel') -> float:
        """计算路径损耗 (dB) - Log-distance模型"""
        if distance <= 0:
            return 0
        
        d0 = 1.0  # 参考距离
        
        # 矿道环境参数
        if environment == 'tunnel':
            n = 2.5  # 路径损耗指数（矿道）
            L0 = 40  # 1米处的损耗
        else:
            n = 2.0
            L0 = 40
        
        # Log-distance模型
        path_loss = L0 + 10 * n * np.log10(distance / d0)
        
        # 阴影衰落（随机成分）
        shadow_fading = np.random.normal(0, 4)
        
        return path_loss + shadow_fading
    
    def calculate_rssi(self, distance: float) -> float:
        """计算RSSI"""
        path_loss = self.calculate_path_loss(distance)
        rssi = self.tx_power - path_loss
        return rssi
    
    def is_link_valid(self, distance: float) -> bool:
        """判断链路是否有效"""
        rssi = self.calculate_rssi(distance)
        return rssi > self.sensitivity
    
    def calculate_throughput(self, distance: float) -> float:
        """根据距离计算吞吐量 (Mbps)"""
        rssi = self.calculate_rssi(distance)
        snr = rssi - (-90)  # 噪声基底
        
        if snr < 10:
            return 0.5
        elif snr < 20:
            return 10.0
        elif snr < 30:
            return 25.0
        else:
            return 40.0


class GraphAttentionDQN(nn.Module):
    """图注意力深度Q网络 - 适用于大规模网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(GraphAttentionDQN, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Q值预测层
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 如果是单个样本，添加batch维度
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 自注意力
        features = features.unsqueeze(1)  # [batch, 1, hidden]
        attn_output, _ = self.attention(features, features, features)
        attn_output = attn_output.squeeze(1)  # [batch, hidden]
        
        # Q值预测
        q_values = self.q_network(attn_output)
        
        # 如果原始输入是1D，返回1D
        if x.dim() == 1:
            q_values = q_values.squeeze(0)
        
        return q_values


class LargeScalePathPlanningAgent:
    """大规模路径规划智能体"""
    
    def __init__(self, topology: LargeScaleMineTopology, wifi_model: WiFiHaLowModel):
        self.topology = topology
        self.wifi_model = wifi_model
        
        # 状态维度：简化表示以处理大规模网络
        # 使用相对位置、目标方向、局部邻居信息
        self.state_dim = 64  # 固定维度的状态表示
        self.action_dim = len(topology.node_list)
        
        # 创建Q网络和目标网络
        self.q_network = GraphAttentionDQN(self.state_dim, self.action_dim, hidden_dim=256)
        self.target_network = GraphAttentionDQN(self.state_dim, self.action_dim, hidden_dim=256)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        
        # 经验回放
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 128
        
        # 训练参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
    def get_state(self, current_node: str, target_node: str, visited: set) -> torch.Tensor:
        """构建状态表示 - 使用特征工程"""
        state = np.zeros(self.state_dim)
        
        # 1. 当前位置特征 (16维)
        current_x, current_y = self.topology.nodes[current_node]
        target_x, target_y = self.topology.nodes[target_node]
        
        # 相对位置
        state[0] = (target_x - current_x) / self.topology.main_tunnel_length
        state[1] = (target_y - current_y) / self.topology.branch_tunnel_length
        
        # 到目标的距离
        dist_to_target = self.topology.get_distance(current_node, target_node)
        state[2] = dist_to_target / self.topology.main_tunnel_length
        
        # 当前节点类型
        state[3] = 1.0 if current_node in self.topology.ap_nodes else 0.0
        state[4] = 1.0 if current_node in self.topology.sta_nodes else 0.0
        state[5] = 1.0 if 'Main' in current_node else 0.0
        state[6] = 1.0 if 'Branch' in current_node else 0.0
        
        # 归一化坐标
        state[7] = current_x / self.topology.main_tunnel_length
        state[8] = current_y / self.topology.branch_tunnel_length
        state[9] = target_x / self.topology.main_tunnel_length
        state[10] = target_y / self.topology.branch_tunnel_length
        
        # 2. 局部邻居信息 (24维)
        neighbors = self.topology.adjacency[current_node]
        for i, (neighbor, dist) in enumerate(neighbors[:6]):  # 最多6个邻居
            base_idx = 11 + i * 4
            if base_idx + 3 < self.state_dim:
                # 邻居相对位置
                n_x, n_y = self.topology.nodes[neighbor]
                state[base_idx] = (n_x - current_x) / 100.0
                state[base_idx + 1] = (n_y - current_y) / 100.0
                # 距离
                state[base_idx + 2] = dist / 100.0
                # 是否访问过
                state[base_idx + 3] = 1.0 if neighbor in visited else 0.0
        
        # 3. 路径统计信息 (10维)
        state[35] = len(visited) / len(self.topology.nodes)  # 访问比例
        
        # 到Portal的大致方向
        portal_x, portal_y = self.topology.nodes['Portal']
        state[36] = (portal_x - current_x) / self.topology.main_tunnel_length
        state[37] = (portal_y - current_y) / self.topology.branch_tunnel_length
        
        # 4. 信号质量特征 (剩余维度)
        # 到最近AP的距离
        min_ap_dist = min([self.topology.get_distance(current_node, ap) 
                          for ap in self.topology.ap_nodes], default=1000)
        state[38] = min_ap_dist / 100.0
        
        return torch.FloatTensor(state)
    
    def select_action(self, state: torch.Tensor, current_node: str, 
                     valid_actions: List[str], training: bool = True) -> str:
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                
                # 只考虑有效动作
                valid_mask = torch.full((self.action_dim,), float('-inf'))
                for action in valid_actions:
                    action_idx = self.topology.node_to_idx[action]
                    valid_mask[action_idx] = q_values[action_idx]
                
                action_idx = torch.argmax(valid_mask).item()
                return self.topology.idx_to_node[action_idx]
    
    def calculate_reward(self, current_node: str, next_node: str, 
                        target_node: str, visited: set, path_length: float) -> float:
        """计算奖励"""
        # 到达目标
        if next_node == target_node:
            return 200.0 - path_length * 0.05
        
        # 重复访问惩罚
        if next_node in visited:
            return -20.0
        
        # AP节点奖励
        reward = 0.0
        if next_node in self.topology.ap_nodes:
            reward += 3.0
        
        # 距离目标的变化
        current_dist = self.topology.get_distance(current_node, target_node)
        next_dist = self.topology.get_distance(next_node, target_node)
        
        if next_dist < current_dist:
            reward += 5.0  # 靠近目标
        else:
            reward -= 2.0  # 远离目标
        
        # 信号质量奖励
        link_dist = self.topology.get_distance(current_node, next_node)
        rssi = self.wifi_model.calculate_rssi(link_dist)
        
        if rssi > -70:
            reward += 4.0
        elif rssi > -85:
            reward += 2.0
        elif rssi > -95:
            reward += 0.5
        else:
            reward -= 3.0
        
        # 路径长度惩罚
        reward -= path_length * 0.01
        
        return reward
    
    def train_step(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor([self.topology.node_to_idx[a] for a in actions])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 损失和优化
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def find_path(self, start_node: str, target_node: str, 
                  max_steps: int = 100) -> Tuple[List[str], float]:
        """寻找路径"""
        path = [start_node]
        visited = {start_node}
        current_node = start_node
        total_distance = 0
        
        for step in range(max_steps):
            valid_neighbors = [neighbor for neighbor, _ in self.topology.adjacency[current_node]
                             if neighbor not in visited]
            
            if not valid_neighbors or current_node == target_node:
                break
            
            state = self.get_state(current_node, target_node, visited)
            next_node = self.select_action(state, current_node, valid_neighbors, training=False)
            
            distance = self.topology.get_distance(current_node, next_node)
            total_distance += distance
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path, total_distance


class LargeScaleTrainer:
    """大规模训练器"""
    
    def __init__(self, agent: LargeScalePathPlanningAgent):
        self.agent = agent
        self.episode_rewards = []
        self.losses = []
        
    def train_episode(self, start_node: str, target_node: str, max_steps: int = 100) -> float:
        """训练一个回合"""
        current_node = start_node
        visited = {start_node}
        total_reward = 0
        path_length = 0
        
        for step in range(max_steps):
            state = self.agent.get_state(current_node, target_node, visited)
            
            valid_actions = [neighbor for neighbor, _ in self.agent.topology.adjacency[current_node]
                           if neighbor not in visited]
            
            if not valid_actions:
                break
            
            action = self.agent.select_action(state, current_node, valid_actions, training=True)
            next_node = action
            distance = self.agent.topology.get_distance(current_node, next_node)
            path_length += distance
            
            reward = self.agent.calculate_reward(current_node, next_node, target_node, 
                                                visited, path_length)
            total_reward += reward
            
            done = (next_node == target_node)
            visited.add(next_node)
            next_state = self.agent.get_state(next_node, target_node, visited)
            
            self.agent.replay_buffer.append((state, action, reward, next_state, done))
            
            loss = self.agent.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            current_node = next_node
            
            if done:
                break
        
        self.agent.epsilon = max(self.agent.epsilon_min, 
                                self.agent.epsilon * self.agent.epsilon_decay)
        
        return total_reward
    
    def train(self, episodes: int = 2000, update_target_every: int = 20):
        """训练"""
        print(f"\n开始训练 {episodes} 个回合...")
        
        for episode in range(episodes):
            # 随机选择起点和终点
            sta_node = random.choice(self.agent.topology.sta_nodes)
            target_node = 'Portal'
            
            total_reward = self.train_episode(sta_node, target_node)
            self.episode_rewards.append(total_reward)
            
            if episode % update_target_every == 0:
                self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        print("训练完成！")


def visualize_large_topology(topology: LargeScaleMineTopology, 
                             path: List[str] = None, 
                             figsize=(20, 12)):
    """可视化大规模拓扑"""
    plt.figure(figsize=figsize)
    
    # 绘制所有边
    for src, dst, _ in topology.edges:
        x1, y1 = topology.nodes[src]
        x2, y2 = topology.nodes[dst]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, linewidth=0.5)
    
    # 绘制节点
    for node, (x, y) in topology.nodes.items():
        if node == 'Portal':
            plt.scatter(x, y, c='green', s=300, marker='*', 
                       label='Portal', zorder=5, edgecolors='black', linewidths=2)
        elif 'STA' in node:
            plt.scatter(x, y, c='red', s=150, marker='^', 
                       label='STA' if node == topology.sta_nodes[0] else '', 
                       zorder=4, edgecolors='black', linewidths=1)
        elif 'AP' in node:
            plt.scatter(x, y, c='blue', s=80, marker='s', 
                       label='AP' if node == 'Main_AP0' else '', 
                       zorder=3, alpha=0.6)
        elif 'Refuge' in node:
            plt.scatter(x, y, c='orange', s=120, marker='D', 
                       label='Refuge' if node == 'Refuge1' else '', 
                       zorder=3)
    
    # 绘制路径
    if path:
        for i in range(len(path) - 1):
            x1, y1 = topology.nodes[path[i]]
            x2, y2 = topology.nodes[path[i + 1]]
            dx, dy = x2 - x1, y2 - y1
            plt.arrow(x1, y1, dx * 0.9, dy * 0.9, 
                     head_width=15, head_length=10, 
                     fc='lime', ec='darkgreen', linewidth=3, zorder=6, alpha=0.8)
    
    plt.xlabel('X 坐标 (米)', fontsize=14)
    plt.ylabel('Y 坐标 (米)', fontsize=14)
    plt.title('大规模矿道WiFi HaLow通信拓扑', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    
    return plt


def visualize_training(trainer: LargeScaleTrainer):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 奖励曲线
    axes[0].plot(trainer.episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
    window = 100
    if len(trainer.episode_rewards) >= window:
        moving_avg = np.convolve(trainer.episode_rewards, 
                                np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(trainer.episode_rewards)), 
                    moving_avg, linewidth=2.5, color='red', 
                    label=f'{window}-episode Moving Avg')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('训练奖励曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    if trainer.losses:
        axes[1].plot(trainer.losses, alpha=0.3, color='orange', linewidth=0.5)
        if len(trainer.losses) >= window:
            moving_avg = np.convolve(trainer.losses, 
                                    np.ones(window) / window, mode='valid')
            axes[1].plot(range(window - 1, len(trainer.losses)), 
                        moving_avg, linewidth=2.5, color='darkred',
                        label=f'{window}-step Moving Avg')
        axes[1].set_xlabel('Training Step', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('训练损失曲线', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def main():
    """主函数"""
    print("=" * 70)
    print(" " * 15 + "大规模WiFi HaLow矿道通信路径规划系统")
    print("=" * 70)
    
    # 创建大规模矿道拓扑
    print("\n正在生成大规模矿道拓扑...")
    topology = LargeScaleMineTopology(
        main_tunnel_length=1200.0,   # 主矿道1200米
        branch_tunnel_length=600.0,  # 分支矿道600米
        ap_spacing=50.0,              # AP间隔50米
        num_branches=3                # 3条分支
    )
    
    info = topology.get_node_info()
    print(f"\n拓扑统计:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 创建WiFi HaLow模型
    wifi_model = WiFiHaLowModel()
    print(f"\nWiFi HaLow参数:")
    print(f"  频率: {wifi_model.frequency} MHz")
    print(f"  发射功率: {wifi_model.tx_power} dBm")
    print(f"  接收灵敏度: {wifi_model.sensitivity} dBm")
    print(f"  最大范围: {wifi_model.max_range} m")
    
    # 创建智能体
    print("\n正在初始化深度强化学习智能体...")
    agent = LargeScalePathPlanningAgent(topology, wifi_model)
    print(f"  状态维度: {agent.state_dim}")
    print(f"  动作维度: {agent.action_dim}")
    print(f"  网络参数量: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # 训练
    trainer = LargeScaleTrainer(agent)
    trainer.train(episodes=2000, update_target_every=20)
    
    # 测试所有STA到Portal的路径
    print("\n" + "=" * 70)
    print("测试路径规划结果")
    print("=" * 70)
    
    all_paths = {}
    for sta_node in topology.sta_nodes:
        print(f"\n{'='*70}")
        print(f"从 {sta_node} 到 Portal 的最优路径:")
        print(f"{'='*70}")
        
        path, distance = agent.find_path(sta_node, 'Portal')
        all_paths[sta_node] = path
        
        print(f"路径长度: {len(path)} 跳")
        print(f"总距离: {distance:.2f} 米")
        print(f"\n路径:")
        print(" → ".join(path[:5]) + " → ...")  # 只显示前5跳
        
        # 计算平均信号质量
        total_rssi = 0
        total_throughput = 0
        for i in range(min(5, len(path) - 1)):  # 只分析前5跳
            link_dist = topology.get_distance(path[i], path[i + 1])
            rssi = wifi_model.calculate_rssi(link_dist)
            throughput = wifi_model.calculate_throughput(link_dist)
            total_rssi += rssi
            total_throughput += throughput
            
            print(f"  [{i+1}] {path[i]} → {path[i + 1]}: "
                  f"距离={link_dist:.1f}m, RSSI={rssi:.1f}dBm, "
                  f"吞吐量={throughput:.1f}Mbps")
        
        if len(path) > 1:
            avg_rssi = total_rssi / min(5, len(path) - 1)
            avg_throughput = total_throughput / min(5, len(path) - 1)
            print(f"\n平均链路质量(前5跳):")
            print(f"  平均RSSI: {avg_rssi:.1f} dBm")
            print(f"  平均吞吐量: {avg_throughput:.1f} Mbps")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 可视化拓扑（选择一个最长的路径）
    longest_path = max(all_paths.values(), key=len)
    plt1 = visualize_large_topology(topology, longest_path)
    plt1.savefig('/mnt/user-data/outputs/large_scale_topology.png', 
                dpi=150, bbox_inches='tight')
    print("✓ 拓扑图已保存")
    
    # 可视化训练结果
    plt2 = visualize_training(trainer)
    plt2.savefig('/mnt/user-data/outputs/training_curves.png', 
                dpi=150, bbox_inches='tight')
    print("✓ 训练曲线已保存")
    
    # 保存模型
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, '/mnt/user-data/outputs/large_scale_model.pth')
    print("✓ 模型已保存")
    
    # 保存拓扑信息
    topology_data = {
        'nodes': {k: list(v) for k, v in topology.nodes.items()},
        'info': info,
        'ap_nodes': topology.ap_nodes,
        'sta_nodes': topology.sta_nodes,
    }
    with open('/mnt/user-data/outputs/topology_data.json', 'w') as f:
        json.dump(topology_data, f, indent=2)
    print("✓ 拓扑数据已保存")
    
    print("\n" + "=" * 70)
    print("程序执行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
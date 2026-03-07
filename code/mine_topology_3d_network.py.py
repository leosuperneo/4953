# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:19:51 2025

@author: HONGTAO LEO
"""

"""
矿道WiFi HaLow通信系统 - 基于PyTorch的路径规划优化
支持深度强化学习（DQN）进行动态路径规划
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json


class MineTopology:
    """矿道拓扑结构建模"""
    
    def __init__(self, main_shaft_depth: int = 500, 
                 num_tunnels: int = 5, 
                 tunnel_length: int = 200,
                 ap_interval: int = 25):
        """
        初始化矿道结构
        :param main_shaft_depth: 主井道深度（米）
        :param num_tunnels: 矿道数量
        :param tunnel_length: 每条矿道长度（米）
        :param ap_interval: AP间隔距离（米）
        """
        self.main_shaft_depth = main_shaft_depth
        self.num_tunnels = num_tunnels
        self.tunnel_length = tunnel_length
        self.ap_interval = ap_interval
        
        self.aps = []  # AP列表
        self.adjacency_matrix = None  # 邻接矩阵
        self._build_topology()
    
    def _build_topology(self):
        """构建矿道AP拓扑"""
        ap_id = 0
        
        # 地面洞口AP（ID=0）
        self.aps.append({
            'id': 0,
            'type': 'entrance',
            'position': (0, 0, 0),  # (x, y, z)
            'name': '洞口基站'
        })
        ap_id += 1
        
        # 主井道AP
        main_shaft_aps = int(self.main_shaft_depth / self.ap_interval)
        for i in range(1, main_shaft_aps + 1):
            depth = i * self.ap_interval
            self.aps.append({
                'id': ap_id,
                'type': 'main_shaft',
                'position': (0, 0, -depth),
                'name': f'主井道-{depth}m'
            })
            ap_id += 1
        
        # 各条矿道AP
        tunnel_depths = np.linspace(100, self.main_shaft_depth, self.num_tunnels)
        for tunnel_idx, depth in enumerate(tunnel_depths):
            num_aps = int(self.tunnel_length / self.ap_interval)
            for i in range(1, num_aps + 1):
                distance = i * self.ap_interval
                self.aps.append({
                    'id': ap_id,
                    'type': 'tunnel',
                    'tunnel_id': tunnel_idx,
                    'position': (distance, tunnel_idx * 30, -depth),
                    'name': f'矿道{tunnel_idx+1}-{distance}m'
                })
                ap_id += 1
        
        self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self):
        """构建AP邻接矩阵（考虑WiFi HaLow的传播特性）"""
        n = len(self.aps)
        self.adjacency_matrix = np.zeros((n, n))
        
        for i, ap1 in enumerate(self.aps):
            for j, ap2 in enumerate(self.aps):
                if i == j:
                    continue
                
                # 计算3D距离
                pos1 = np.array(ap1['position'])
                pos2 = np.array(ap2['position'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # WiFi HaLow在矿道的有效通信距离判断
                max_distance = 100  # 考虑矿道环境，最大通信距离100m
                
                if distance <= max_distance:
                    # 信号强度模型（简化的对数距离路径损耗模型）
                    # RSSI = RSSI_0 - 10 * n * log10(d/d0) - 额外损耗
                    rssi_0 = -30  # 1米处参考信号强度
                    path_loss_exponent = 3.5  # 矿道环境路径损耗指数
                    extra_loss = 15  # 矿道额外损耗
                    
                    if distance > 0:
                        rssi = rssi_0 - 10 * path_loss_exponent * np.log10(distance) - extra_loss
                    else:
                        rssi = rssi_0
                    
                    # 将RSSI转换为链路质量（0-1）
                    # RSSI范围假设为 -90dBm 到 -30dBm
                    link_quality = np.clip((rssi + 90) / 60, 0, 1)
                    
                    self.adjacency_matrix[i][j] = link_quality
    
    def get_state_size(self):
        """获取状态空间大小"""
        return len(self.aps)
    
    def get_action_size(self):
        """获取动作空间大小（可选择的下一跳AP数量）"""
        return len(self.aps)
    
    def visualize(self):
        """可视化矿道拓扑"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制AP
        for ap in self.aps:
            pos = ap['position']
            if ap['type'] == 'entrance':
                ax.scatter(*pos, c='red', s=200, marker='*', label='洞口')
            elif ap['type'] == 'main_shaft':
                ax.scatter(*pos, c='blue', s=100, marker='o', alpha=0.6)
            else:
                ax.scatter(*pos, c='green', s=80, marker='^', alpha=0.6)
        
        # 绘制连接
        n = len(self.aps)
        for i in range(n):
            for j in range(i+1, n):
                if self.adjacency_matrix[i][j] > 0.3:  # 只显示强连接
                    pos1 = self.aps[i]['position']
                    pos2 = self.aps[j]['position']
                    ax.plot([pos1[0], pos2[0]], 
                           [pos1[1], pos2[1]], 
                           [pos1[2], pos2[2]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('深度 (m)')
        ax.set_title('矿道WiFi HaLow AP拓扑结构')
        plt.savefig('/mnt/user-data/outputs/mine_topology.png', dpi=300, bbox_inches='tight')
        print("拓扑图已保存")


class DQNNetwork(nn.Module):
    """深度Q网络用于路径规划"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class CommunicationEnvironment:
    """矿道通信环境"""
    
    def __init__(self, topology: MineTopology):
        self.topology = topology
        self.current_ap = None
        self.target_ap = 0  # 目标总是洞口（ID=0）
        self.visited = set()
        self.path = []
        self.total_hops = 0
        self.total_quality = 0.0
    
    def reset(self, start_ap: int = None):
        """重置环境，随机选择一个起始AP"""
        if start_ap is None:
            # 从深处矿道随机选择起始点
            tunnel_aps = [ap for ap in self.topology.aps if ap['type'] == 'tunnel']
            start = random.choice(tunnel_aps)
            self.current_ap = start['id']
        else:
            self.current_ap = start_ap
        
        self.visited = {self.current_ap}
        self.path = [self.current_ap]
        self.total_hops = 0
        self.total_quality = 0.0
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        n = len(self.topology.aps)
        state = np.zeros(n)
        
        # 当前位置
        state[self.current_ap] = 1.0
        
        # 目标位置
        state[self.target_ap] = 0.5
        
        # 邻居信号质量
        neighbors = self.topology.adjacency_matrix[self.current_ap]
        state += neighbors * 0.3
        
        return state
    
    def step(self, action: int):
        """执行动作（选择下一跳AP）"""
        # 检查动作有效性
        link_quality = self.topology.adjacency_matrix[self.current_ap][action]
        
        if link_quality == 0:
            # 无效动作（不可达）
            reward = -10
            done = False
            return self._get_state(), reward, done, {}
        
        # 计算奖励
        reward = 0
        
        # 链路质量奖励
        reward += link_quality * 2
        
        # 接近目标奖励
        current_dist = np.linalg.norm(
            np.array(self.topology.aps[self.current_ap]['position']) - 
            np.array(self.topology.aps[self.target_ap]['position'])
        )
        next_dist = np.linalg.norm(
            np.array(self.topology.aps[action]['position']) - 
            np.array(self.topology.aps[self.target_ap]['position'])
        )
        
        if next_dist < current_dist:
            reward += 5
        else:
            reward -= 2
        
        # 重复访问惩罚
        if action in self.visited:
            reward -= 5
        
        # 跳数惩罚
        reward -= 0.5
        
        # 执行动作
        self.current_ap = action
        self.visited.add(action)
        self.path.append(action)
        self.total_hops += 1
        self.total_quality += link_quality
        
        # 检查是否到达目标
        done = (action == self.target_ap)
        if done:
            reward += 50  # 到达目标大奖励
            # 额外奖励：路径短且质量高
            avg_quality = self.total_quality / self.total_hops if self.total_hops > 0 else 0
            reward += avg_quality * 10
            reward -= self.total_hops * 0.5
        
        # 超过最大跳数
        if self.total_hops > 20:
            done = True
            reward -= 20
        
        return self._get_state(), reward, done, {
            'path': self.path.copy(),
            'hops': self.total_hops,
            'avg_quality': self.total_quality / self.total_hops if self.total_hops > 0 else 0
        }
    
    def get_valid_actions(self):
        """获取当前可用的动作"""
        valid = []
        for i, quality in enumerate(self.topology.adjacency_matrix[self.current_ap]):
            if quality > 0:
                valid.append(i)
        return valid


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        
        # 主网络和目标网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """选择动作（ε-greedy策略）"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # 只从有效动作中选择
        valid_q_values = [(a, q_values[a]) for a in valid_actions]
        return max(valid_q_values, key=lambda x: x[1])[0]
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # 计算损失
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_agent(episodes: int = 1000, save_interval: int = 100):
    """训练DQN智能体"""
    
    # 创建环境
    topology = MineTopology(
        main_shaft_depth=500,
        num_tunnels=5,
        tunnel_length=200,
        ap_interval=25
    )
    
    env = CommunicationEnvironment(topology)
    agent = DQNAgent(
        state_size=topology.get_state_size(),
        action_size=topology.get_action_size()
    )
    
    # 训练统计
    episode_rewards = []
    episode_hops = []
    success_rate = []
    losses = []
    
    print(f"开始训练... 总AP数: {len(topology.aps)}")
    print(f"设备: {agent.device}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        
        while True:
            # 选择动作
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 更新统计
        episode_rewards.append(total_reward)
        episode_hops.append(info['hops'])
        
        # 计算最近100局的成功率
        recent_success = sum(1 for i in range(max(0, episode-99), episode+1) 
                            if episode_hops[i] < 20) / min(episode+1, 100)
        success_rate.append(recent_success)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # 更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_hops = np.mean(episode_hops[-50:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Hops: {avg_hops:.2f} | "
                  f"Success Rate: {recent_success:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(f'/home/claude/dqn_model_ep{episode+1}.pth')
    
    # 保存最终模型
    agent.save('/mnt/user-data/outputs/dqn_model_final.pth')
    
    # 绘制训练曲线
    plot_training_results(episode_rewards, episode_hops, success_rate, losses)
    
    return agent, topology, env


def plot_training_results(rewards, hops, success_rate, losses):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward')
    axes[0, 0].plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), 
                    label='Moving Average (50)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('训练奖励曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 跳数曲线
    axes[0, 1].plot(hops, alpha=0.6, label='Episode Hops')
    axes[0, 1].plot(np.convolve(hops, np.ones(50)/50, mode='valid'), 
                    label='Moving Average (50)', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Number of Hops')
    axes[0, 1].set_title('路径跳数曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 成功率曲线
    axes[1, 0].plot(success_rate, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('成功率曲线（最近100局）')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])
    
    # 损失曲线
    if losses:
        axes[1, 1].plot(losses, alpha=0.6)
        axes[1, 1].plot(np.convolve(losses, np.ones(50)/50, mode='valid'), 
                        linewidth=2, color='red')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('训练损失曲线')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_results.png', dpi=300, bbox_inches='tight')
    print("训练结果图已保存")


def test_agent(agent, topology, env, num_tests: int = 20):
    """测试训练好的智能体"""
    print("\n开始测试智能体性能...")
    
    agent.epsilon = 0  # 关闭探索
    
    test_results = []
    
    for test_id in range(num_tests):
        # 随机选择深处矿道的起始点
        tunnel_aps = [ap for ap in topology.aps if ap['type'] == 'tunnel']
        start_ap = random.choice(tunnel_aps)['id']
        
        state = env.reset(start_ap)
        path = [start_ap]
        total_hops = 0
        success = False
        
        while total_hops < 20:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_hops += 1
            
            if done and action == 0:  # 成功到达洞口
                success = True
                break
        
        # 构建路径描述
        path_desc = " -> ".join([topology.aps[ap_id]['name'] for ap_id in info['path']])
        
        result = {
            'test_id': test_id + 1,
            'start_ap': topology.aps[start_ap]['name'],
            'success': success,
            'hops': info['hops'],
            'avg_quality': info['avg_quality'],
            'path': info['path'],
            'path_description': path_desc
        }
        
        test_results.append(result)
        
        status = "✓ 成功" if success else "✗ 失败"
        print(f"\n测试 {test_id+1}: {status}")
        print(f"  起点: {result['start_ap']}")
        print(f"  跳数: {result['hops']}")
        print(f"  平均链路质量: {result['avg_quality']:.3f}")
        print(f"  路径: {path_desc}")
    
    # 统计结果
    success_count = sum(1 for r in test_results if r['success'])
    avg_hops = np.mean([r['hops'] for r in test_results if r['success']])
    avg_quality = np.mean([r['avg_quality'] for r in test_results if r['success']])
    
    print(f"\n{'='*60}")
    print(f"测试总结:")
    print(f"  总测试次数: {num_tests}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_count/num_tests:.1%}")
    print(f"  平均跳数: {avg_hops:.2f}")
    print(f"  平均链路质量: {avg_quality:.3f}")
    print(f"{'='*60}")
    
    # 保存测试结果
    with open('/mnt/user-data/outputs/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    return test_results


def visualize_path(topology, path, filename='communication_path.png'):
    """可视化通信路径"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有AP（半透明）
    for ap in topology.aps:
        pos = ap['position']
        if ap['type'] == 'entrance':
            ax.scatter(*pos, c='red', s=200, marker='*', alpha=0.3)
        elif ap['type'] == 'main_shaft':
            ax.scatter(*pos, c='blue', s=100, marker='o', alpha=0.2)
        else:
            ax.scatter(*pos, c='green', s=80, marker='^', alpha=0.2)
    
    # 高亮路径上的AP
    for ap_id in path:
        pos = topology.aps[ap_id]['position']
        if ap_id == 0:
            ax.scatter(*pos, c='red', s=300, marker='*', label='洞口', edgecolors='black', linewidths=2)
        elif ap_id == path[0]:
            ax.scatter(*pos, c='orange', s=200, marker='D', label='起点', edgecolors='black', linewidths=2)
        else:
            ax.scatter(*pos, c='yellow', s=150, marker='o', edgecolors='black', linewidths=1.5)
    
    # 绘制路径
    for i in range(len(path) - 1):
        pos1 = topology.aps[path[i]]['position']
        pos2 = topology.aps[path[i+1]]['position']
        ax.plot([pos1[0], pos2[0]], 
               [pos1[1], pos2[1]], 
               [pos1[2], pos2[2]], 
               'r-', linewidth=3, alpha=0.8, label='通信路径' if i == 0 else '')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('深度 (m)')
    ax.set_title('WiFi HaLow最优通信路径')
    ax.legend()
    
    plt.savefig(f'/mnt/user-data/outputs/{filename}', dpi=300, bbox_inches='tight')
    print(f"路径可视化已保存: {filename}")


if __name__ == "__main__":
    print("="*70)
    print("矿道WiFi HaLow通信系统 - PyTorch路径规划优化")
    print("="*70)
    
    # 1. 创建并可视化矿道拓扑
    print("\n[步骤 1] 构建矿道拓扑结构...")
    topology = MineTopology(
        main_shaft_depth=500,
        num_tunnels=5,
        tunnel_length=200,
        ap_interval=25
    )
    print(f"✓ 拓扑构建完成: 共 {len(topology.aps)} 个AP节点")
    topology.visualize()
    
    # 2. 训练DQN智能体
    print("\n[步骤 2] 开始训练DQN智能体...")
    agent, topology, env = train_agent(episodes=500, save_interval=100)
    print("✓ 训练完成")
    
    # 3. 测试智能体
    print("\n[步骤 3] 测试训练好的智能体...")
    test_results = test_agent(agent, topology, env, num_tests=10)
    
    # 4. 可视化最佳路径
    print("\n[步骤 4] 可视化通信路径...")
    successful_tests = [r for r in test_results if r['success']]
    if successful_tests:
        # 选择跳数最少的路径
        best_test = min(successful_tests, key=lambda x: x['hops'])
        visualize_path(topology, best_test['path'], 'best_communication_path.png')
        print(f"✓ 最佳路径: {best_test['hops']} 跳")
    
    print("\n" + "="*70)
    print("所有任务完成！结果已保存到 /mnt/user-data/outputs/")
    print("="*70)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Dict
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建输出目录
os.makedirs('/mnt/user-data/outputs', exist_ok=True)


class MineEnvironment:
    """矿道环境模拟器"""
    
    def __init__(self, start_ap_idx=0, target_ap_idx=None, branch_config=None):
        """
        初始化矿道环境
        
        参数:
            start_ap_idx: 起始AP索引，默认为0（最左侧）
            target_ap_idx: 目标AP索引，默认为None（自动选择最右侧主矿道AP）
            branch_config: 分支配置列表，每个元素为字典，格式：
                {
                    'x': 分支在主矿道的位置(米),
                    'has_up': 是否有向上的分支(bool),
                    'has_down': 是否有向下的分支(bool),
                    'length': 分支长度(米)
                }
                
                示例:
                # 2个双向分支（默认）
                [
                    {'x': 200, 'has_up': True, 'has_down': True, 'length': 600},
                    {'x': 700, 'has_up': True, 'has_down': True, 'length': 600}
                ]
                
                # 4个分支，不同方向
                [
                    {'x': 150, 'has_up': True, 'has_down': True, 'length': 400},
                    {'x': 400, 'has_up': True, 'has_down': False, 'length': 500},
                    {'x': 650, 'has_up': False, 'has_down': True, 'length': 300},
                    {'x': 900, 'has_up': True, 'has_down': True, 'length': 600}
                ]
                
                如果为None，使用默认的2个双向分支配置
        """
        # 矿道结构参数
        self.main_tunnel_length = 1200  # 主矿道长度(米)
        self.branch_tunnel_length = 600  # 默认分支矿道长度(米)
        self.ap_interval = 50  # AP间隔(米)
        # === 拓扑传播参数（新增） ===
        self.bend_loss_db = 25.0      # 每个90°转弯附加损耗(dB)
        self.block_loss_db = 110.0     # 非矿道穿墙的硬阻断(dB)
        self.rssi_vis_thresh = -90.0   # 候选AP最小RSSI阈值(dBm)

        
        # 设置分支配置
        if branch_config is None:
            # 默认配置：2个双向分支
            self.branch_config = [
                {'x': 200, 'has_up': True, 'has_down': True, 'length': 600},
                {'x': 700, 'has_up': True, 'has_down': True, 'length': 600}
            ]
        else:
            self.branch_config = branch_config
        
        print(f"\n{'='*60}")
        print(f"矿道配置")
        print(f"{'='*60}")
        print(f"主矿道: 长度 {self.main_tunnel_length}米, AP间隔 {self.ap_interval}米")
        print(f"分支数量: {len(self.branch_config)}个")
        for i, branch in enumerate(self.branch_config):
            direction = []
            if branch.get('has_up', True):
                direction.append('↑上')
            if branch.get('has_down', True):
                direction.append('↓下')
            print(f"  分支{i+1}: x={branch['x']}米, 长度={branch.get('length', 600)}米, 方向={'+'.join(direction)}")
        print(f"{'='*60}\n")
        
        # 创建AP布局
        self.aps = self._create_ap_layout()
        self.num_aps = len(self.aps)
        
        # WiFi HaLow 参数
        self.freq = 900e6  # 900 MHz
        self.tx_power = 30  # dBm 发射功率
        self.noise_floor = -95  # dBm 噪声底
        self.max_range = 1000  # 最大通信距离(米)
        
        # 保存起始位置索引
        self.start_ap_idx = start_ap_idx
        
        # 状态参数
        self.current_ap_idx = start_ap_idx  # 当前连接的AP索引
        
        # 设置目标AP
        if target_ap_idx is not None:
            self.target_ap_idx = target_ap_idx
        else:
            self._find_target_ap()  # 自动找到最右侧的AP作为目标
        
        self.position = np.array(self.aps[self.start_ap_idx][:2], dtype=float)  # 当前STA位置
        self.target_position = np.array(self.aps[self.target_ap_idx][:2], dtype=float)
        
        print(f"起始AP: {self.aps[self.start_ap_idx][2]} at ({self.position[0]}, {self.position[1]})")
        print(f"目标AP: {self.aps[self.target_ap_idx][2]} at ({self.target_position[0]}, {self.target_position[1]})")
        
        # 统计信息
        self.steps = 0
        self.max_steps = 200
        self.handover_count = 0
        self.total_signal_quality = 0
        
        # 统计信息
        self.steps = 0
        self.max_steps = 200
        self.handover_count = 0
        self.total_signal_quality = 0
        
    def _create_ap_layout(self) -> List[Tuple[float, float, str]]:
        """创建AP布局 - 返回[(x, y, name), ...]"""
        aps = []
        ap_id = 0
        
        # 主矿道APs (从左到右, y=0)
        num_main_aps = int(self.main_tunnel_length / self.ap_interval) + 1
        for i in range(num_main_aps):
            x = i * self.ap_interval
            y = 0
            aps.append((x, y, f"MainAP{ap_id}"))
            ap_id += 1
        
        print(f"主矿道AP数量: {num_main_aps}")
        
        # 根据配置创建分支矿道APs
        for branch_idx, branch in enumerate(self.branch_config):
            branch_x = branch['x']
            branch_length = branch.get('length', self.branch_tunnel_length)
            has_up = branch.get('has_up', True)
            has_down = branch.get('has_down', True)
            
            num_branch_aps = int(branch_length / self.ap_interval)
            
            print(f"分支{branch_idx+1} at x={branch_x}m: ", end="")
            
            # 上分支
            if has_up:
                for i in range(1, num_branch_aps + 1):
                    x = branch_x
                    y = i * self.ap_interval
                    aps.append((x, y, f"Branch{branch_idx+1}_Up_AP{ap_id}"))
                    ap_id += 1
                print(f"上分支{num_branch_aps}个AP", end=" ")
            
            # 下分支
            if has_down:
                for i in range(1, num_branch_aps + 1):
                    x = branch_x
                    y = -i * self.ap_interval
                    aps.append((x, y, f"Branch{branch_idx+1}_Down_AP{ap_id}"))
                    ap_id += 1
                print(f"下分支{num_branch_aps}个AP", end="")
            
            print()
        
        print(f"总共创建了 {len(aps)} 个AP")
        return aps
    

    # ====== 拓扑辅助函数（新增） ======
    def _branch_axes(self):
        """根据 branch_config 返回有效竖巷轴与其半长：[(x, half_len, has_up, has_down), ...]"""
        axes = []
        for b in self.branch_config:
            x = float(b['x'])
            L = float(b.get('length', self.branch_tunnel_length))
            half = L
            axes.append((x, half, bool(b.get('has_up', True)), bool(b.get('has_down', True))))
        return axes

    def _on_corridor(self, p: tuple) -> bool:
        """点是否在矿道网络线上（主巷 y=0 或启用的竖巷）"""
        x, y = float(p[0]), float(p[1])
        if abs(y) < 1e-6 and -1e-6 <= x <= self.main_tunnel_length + 1e-6:
            return True
        for bx, half, up, down in self._branch_axes():
            if abs(x - bx) < 1e-6:
                if (up and 0 - 1e-6 <= y <= half + 1e-6) or (down and -half - 1e-6 <= y <= 0 + 1e-6):
                    return True
        return False

    def _corridor_distance_and_bends(self, p: tuple, q: tuple):
        """沿矿道网络的路径距离与拐角次数。若任一点不在网络上，返回 (inf, 0)。"""
        x1, y1 = float(p[0]), float(p[1])
        x2, y2 = float(q[0]), float(q[1])
        if not self._on_corridor((x1, y1)) or not self._on_corridor((x2, y2)):
            return float('inf'), 0
        if abs(y1) < 1e-6 and abs(y2) < 1e-6:
            return abs(x1 - x2), 0
        for bx, _, up, down in self._branch_axes():
            if abs(x1 - bx) < 1e-6 and abs(x2 - bx) < 1e-6:
                return abs(y1 - y2), 0
        d = 0.0; bends = 0
        if abs(y1) >= 1e-6:
            d += abs(y1); bends += 1
        d += abs(x1 - x2)
        if abs(y2) >= 1e-6:
            d += abs(y2); bends += 1
        return d, bends

    def _visible_on_corridor(self, p: tuple, q: tuple) -> bool:
        d, _ = self._corridor_distance_and_bends(p, q)
        return np.isfinite(d)

    def _project_to_corridor(self, p: tuple) -> tuple:
        """将任意点吸附到最近的矿道线上（L1）"""
        x, y = float(p[0]), float(p[1])
        cands = [(min(max(0.0, x), self.main_tunnel_length), 0.0)]
        for bx, half, up, down in self._branch_axes():
            if up:
                cands.append((bx, min(max(y, 0.0), half)))
            if down:
                cands.append((bx, max(min(y, 0.0), -half)))
        best = min(cands, key=lambda c: abs(c[0]-x)+abs(c[1]-y))
        return best

    def _move_along_corridor(self, step_size: float = 10.0):
        """沿矿道网络向目标移动，每步 step_size 米"""
        x, y = float(self.position[0]), float(self.position[1])
        tx, ty = float(self.target_position[0]), float(self.target_position[1])
        if not self._on_corridor((x, y)):
            x, y = self._project_to_corridor((x, y))
        if abs(y) < 1e-6:
            if abs(x - tx) > 1e-6:
                dx = np.sign(tx - x) * min(step_size, abs(tx - x))
                x += dx
            else:
                if abs(ty) >= 1e-6:
                    dy = np.sign(ty - y) * min(step_size, abs(ty - y))
                    y += dy
        else:
            if abs(ty) < 1e-6:
                dy = -np.sign(y) * min(step_size, abs(y))
                y += dy
            else:
                if abs(x - tx) < 1e-6:
                    dy = np.sign(ty - y) * min(step_size, abs(ty - y))
                    y += dy
                else:
                    dy = -np.sign(y) * min(step_size, abs(y))
                    y += dy
        self.position = np.array([x, y], dtype=float)
    def _find_target_ap(self):
        """找到最右侧主矿道的AP作为目标"""
        max_x = -1
        for i, ap in enumerate(self.aps):
            if ap[1] == 0 and ap[0] > max_x:  # 主矿道上(y=0)且x最大
                max_x = ap[0]
                self.target_ap_idx = i
        self.target_position = np.array(self.aps[self.target_ap_idx][:2], dtype=float)
        print(f"目标AP: {self.aps[self.target_ap_idx][2]} at ({self.target_position[0]}, {self.target_position[1]})")
    
    def _calculate_path_loss(self, distance: float) -> float:
        """计算路径损耗 (Log-distance path loss model)"""
        if distance < 1:
            distance = 1
        # WiFi HaLow在矿道环境中的路径损耗模型
        d0 = 1.0  # 参考距离
        n = 3.5  # 路径损耗指数（矿道环境较大）
        X_sigma = np.random.normal(0, 4)  # 阴影衰落
        
        pl_d0 = 20 * np.log10(self.freq) + 20 * np.log10(d0) - 147.55
        path_loss = pl_d0 + 10 * n * np.log10(distance / d0) + X_sigma
        
        return path_loss
    
    def _calculate_rssi(self, ap_idx: int) -> float:
        ap_pos = np.array(self.aps[ap_idx][:2], dtype=float)
        d_corr, n_bends = self._corridor_distance_and_bends(tuple(self.position), tuple(ap_pos))
        if not np.isfinite(d_corr) or d_corr > self.max_range:
            return -120.0
        d0 = 1.0
        n = 2.0
        X_sigma = np.random.normal(0, 3.0)
        pl_d0 = 20 * np.log10(self.freq) + 20 * np.log10(d0) - 147.55
        path_loss = pl_d0 + 10 * n * np.log10(max(d_corr, d0) / d0) + X_sigma
        path_loss += n_bends * self.bend_loss_db
        if not self._visible_on_corridor(tuple(self.position), tuple(ap_pos)):
            path_loss += self.block_loss_db
        rssi = self.tx_power - path_loss

        
        return rssi
    
    def _calculate_sinr(self, ap_idx: int) -> float:
        """计算SINR (信号与干扰加噪声比)"""
        rssi = self._calculate_rssi(ap_idx)
        
        # 计算来自其他AP的干扰
        interference = 0
        for i, ap in enumerate(self.aps):
            if i != ap_idx:
                other_rssi = self._calculate_rssi(i)
                if other_rssi > -90:  # 只考虑强干扰
                    interference += 10 ** (other_rssi / 10)
        
        noise = 10 ** (self.noise_floor / 10)
        signal = 10 ** (rssi / 10)
        
        if signal <= 0:
            return -50
        
        sinr = 10 * np.log10(signal / (interference + noise + 1e-10))
        return sinr
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态向量（包含AP物理位置信息作为权重）"""
        # 状态包含：
        # 1. 当前STA位置相对于目标的归一化坐标 (2维)
        # 2. 当前AP的物理位置（归一化） (2维)
        # 3. 当前AP相对于目标AP的位置向量 (2维)
        # 4. 当前AP的SINR (1维)
        # 5. Top-K可达AP的RSSI (10维)
        # 6. Top-K可达AP的相对位置（相对于当前位置） (10×2=20维)
        # 7. Top-K可达AP距离目标的距离 (10维)
        # 8. Top-K可达AP的绝对位置（归一化） (10×2=20维)
        # 9. 切换次数的归一化值 (1维)
        # 总计: 2+2+2+1+10+20+10+20+1 = 68维
        
        # 1. 当前STA位置相对于目标的归一化坐标
        rel_pos = (self.target_position - self.position) / self.main_tunnel_length
        
        # 2. 当前AP的物理位置（归一化）
        current_ap_pos = np.array(self.aps[self.current_ap_idx][:2])
        current_ap_pos_norm = current_ap_pos / np.array([self.main_tunnel_length, self.branch_tunnel_length])
        
        # 3. 当前AP相对于目标AP的位置向量
        target_ap_pos = np.array(self.aps[self.target_ap_idx][:2])
        current_ap_to_target = (target_ap_pos - current_ap_pos) / self.main_tunnel_length
        
        # 4. 当前AP的SINR
        current_sinr = np.clip(self._calculate_sinr(self.current_ap_idx) / 50, -1, 1)  # 归一化并裁剪
        
        # 5-8. 获取Top-K可达AP的信息
        k = 10
        ap_info = []
        for i in range(self.num_aps):
            rssi = self._calculate_rssi(i)
            if rssi > -90 and self._visible_on_corridor(tuple(self.position), tuple(self.aps[i][:2])):  # 只考虑可见且信号较强的AP
                ap_pos = np.array(self.aps[i][:2])
                dist_to_target = np.linalg.norm(ap_pos - target_ap_pos)
                ap_info.append({
                    'idx': i,
                    'rssi': rssi,
                    'pos': ap_pos,
                    'dist_to_target': dist_to_target
                })
        
        # 按RSSI排序，取Top-K
        ap_info.sort(key=lambda x: x['rssi'], reverse=True)
        top_k_aps = ap_info[:k]
        
        # 提取Top-K AP的特征
        top_k_rssi = []
        top_k_rel_pos = []
        top_k_dist_to_target = []
        top_k_abs_pos = []
        
        for ap in top_k_aps:
            # RSSI归一化
            top_k_rssi.append((ap['rssi'] + 100) / 100)
            
            # 相对位置（相对于当前STA位置）- 这个很重要，帮助网络理解方向
            rel_ap_pos = (ap['pos'] - self.position) / self.main_tunnel_length
            top_k_rel_pos.extend(rel_ap_pos)
            
            # 距离目标的距离（归一化）- 帮助网络选择更接近目标的AP
            top_k_dist_to_target.append(ap['dist_to_target'] / self.main_tunnel_length)
            
            # AP的绝对位置（归一化）- 帮助网络理解整个矿道布局
            abs_pos_norm = ap['pos'] / np.array([self.main_tunnel_length, self.branch_tunnel_length])
            top_k_abs_pos.extend(abs_pos_norm)
        
        # 补齐到K个
        while len(top_k_rssi) < k:
            top_k_rssi.append(0)
            top_k_rel_pos.extend([0, 0])
            top_k_dist_to_target.append(1.0)  # 最大距离
            top_k_abs_pos.extend([0, 0])
        
        # 9. 切换次数归一化
        handover_norm = min(self.handover_count / 20, 1.0)
        
        # 组合所有特征
        state = np.concatenate([
            rel_pos,                    # 2
            current_ap_pos_norm,        # 2
            current_ap_to_target,       # 2
            [current_sinr],             # 1
            top_k_rssi,                 # 10
            top_k_rel_pos,              # 20
            top_k_dist_to_target,       # 10
            top_k_abs_pos,              # 20
            [handover_norm]             # 1
        ])
        
        return state.astype(np.float32)
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_ap_idx = self.start_ap_idx
        self.position = np.array(self.aps[self.start_ap_idx][:2], dtype=float)
        self.steps = 0
        self.handover_count = 0
        self.total_signal_quality = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作
        action: 0=保持当前AP, 1-9=切换到候选AP 1-9
        """
        self.steps += 1
        reward = 0
        done = False
        info = {}
        
        # 获取可达AP列表（RSSI > -85 dBm），按RSSI排序
        reachable_aps = []
        for i in range(self.num_aps):
            rssi = self._calculate_rssi(i)
            if rssi > -85 and self._visible_on_corridor(tuple(self.position), tuple(self.aps[i][:2])):
                reachable_aps.append((i, rssi))
        
        reachable_aps.sort(key=lambda x: x[1], reverse=True)
        
        # 执行动作
        old_ap_idx = self.current_ap_idx
        if action == 0:
            # 保持当前AP
            target_ap = self.current_ap_idx
        else:
            # 切换到候选AP
            if action <= len(reachable_aps):
                target_ap = reachable_aps[action - 1][0]
            else:
                target_ap = self.current_ap_idx  # 无效动作，保持不变
        
        # 切换AP
        if target_ap != old_ap_idx:
            self.handover_count += 1
            reward -= 2  # 切换惩罚
            self.current_ap_idx = target_ap
        
        # 在移动前记录与目标的旧距离
        old_distance_to_target = np.linalg.norm(self.target_position - self.position)

        # STA沿矿道移动
        self._move_along_corridor(step_size=10.0)

        # 计算奖励
        new_distance_to_target = np.linalg.norm(self.target_position - self.position)
        
        # 1. 距离奖励：越接近目标越好
        progress_reward = (old_distance_to_target - new_distance_to_target) * 2
        reward += progress_reward
        
        # 2. 信号质量奖励
        current_sinr = self._calculate_sinr(self.current_ap_idx)
        if current_sinr > 20:
            reward += 3
        elif current_sinr > 10:
            reward += 1
        elif current_sinr < 0:
            reward -= 3  # 信号太差的惩罚
        
        # 3. AP位置奖励：如果当前AP更接近目标，给予奖励
        current_ap_pos = np.array(self.aps[self.current_ap_idx][:2])
        ap_dist_to_target = np.linalg.norm(current_ap_pos - self.target_position)
        
        if ap_dist_to_target < 100:  # 如果AP离目标很近
            reward += 5
        elif ap_dist_to_target < 300:
            reward += 2
        
        self.total_signal_quality += current_sinr
        
        # 检查是否到达目标
        if new_distance_to_target < 20:  # 到达目标附近20米
            reward += 100  # 到达目标的大奖励
            done = True
            info['success'] = True
            print(f"✓ 成功到达目标！总步数: {self.steps}, 切换次数: {self.handover_count}")
        
        # 检查是否超时
        if self.steps >= self.max_steps:
            reward -= 50  # 超时惩罚
            done = True
            info['success'] = False
            print(f"✗ 超时未到达目标。距离: {new_distance_to_target:.1f}m")
        
        info['distance_to_target'] = new_distance_to_target
        info['handover_count'] = self.handover_count
        info['sinr'] = current_sinr
        
        next_state = self._get_state()
        
        return next_state, reward, done, info


class DQNetwork(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DQNetwork, self).__init__()
        
        # 位置特征提取网络（专门处理位置相关的输入）
        # 位置特征: rel_pos(2) + current_ap_pos(2) + ap_to_target(2) + top_k_rel_pos(20) + top_k_abs_pos(20) = 46维
        self.position_net = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 信号特征提取网络（专门处理信号相关的输入）
        # 信号特征: sinr(1) + top_k_rssi(10) + top_k_dist_to_target(10) + handover(1) = 22维
        self.signal_net = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 优势流 (Dueling DQN)
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 价值流 (Dueling DQN)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        # 分离位置特征和信号特征
        # 状态结构: rel_pos(2) + current_ap_pos(2) + ap_to_target(2) + sinr(1) + 
        #          rssi(10) + rel_pos(20) + dist(10) + abs_pos(20) + handover(1) = 68
        
        # 位置特征: [0:2, 2:4, 4:6, 17:37, 47:67] 
        # = rel_pos(2) + current_ap_pos(2) + ap_to_target(2) + top_k_rel_pos(20) + top_k_abs_pos(20) = 46维
        pos_features = torch.cat([
            state[:, 0:2],    # rel_pos
            state[:, 2:4],    # current_ap_pos
            state[:, 4:6],    # ap_to_target
            state[:, 17:37],  # top_k_rel_pos
            state[:, 47:67]   # top_k_abs_pos
        ], dim=1)
        
        # 信号特征: [6:7, 7:17, 37:47, 67:68] 
        # = sinr(1) + rssi(10) + dist_to_target(10) + handover(1) = 22维
        signal_features = torch.cat([
            state[:, 6:7],    # sinr
            state[:, 7:17],   # top_k_rssi
            state[:, 37:47],  # top_k_dist_to_target
            state[:, 67:68]   # handover_norm
        ], dim=1)
        
        # 特征提取
        pos_embed = self.position_net(pos_features)
        signal_embed = self.signal_net(signal_features)
        
        # 融合
        combined = torch.cat([pos_embed, signal_embed], dim=1)
        features = self.fusion_net(combined)
        
        # Dueling DQN
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络
        self.policy_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        # 超参数
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10
        
        # 统计
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def train_step(self):
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 采样batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.training_step += 1
        
        # 更新目标网络
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(num_episodes: int = 1000, start_ap_idx: int = 0, target_ap_idx: int = None, branch_config=None):
    """训练DQN
    
    参数:
        num_episodes: 训练回合数
        start_ap_idx: 起始AP索引（0表示最左侧第一个AP）
        target_ap_idx: 目标AP索引（None表示自动选择最右侧主矿道AP）
        branch_config: 分支配置列表，格式为 [{'x': x坐标, 'has_up': bool, 'has_down': bool, 'length': 长度}, ...]
    """
    env = MineEnvironment(start_ap_idx=start_ap_idx, target_ap_idx=target_ap_idx, branch_config=branch_config)
    
    # 获取状态和动作维度
    state_dim = env._get_state().shape[0]
    action_dim = 10  # 0=保持, 1-9=切换到前9个候选AP
    
    print(f"\n状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"AP总数: {env.num_aps}\n")
    
    agent = DQNAgent(state_dim, action_dim)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    losses = []
    
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        done = False
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # 更新epsilon
        agent.update_epsilon()
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(env.steps)
        
        if info.get('success', False):
            success_count += 1
        
        if len(episode_loss) > 0:
            losses.append(np.mean(episode_loss))
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), '/mnt/user-data/outputs/best_model.pth')
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            success_rate = success_count / (episode + 1) * 100
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均步数: {avg_length:.1f}")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  缓冲区大小: {len(agent.replay_buffer)}")
            if len(losses) > 0:
                print(f"  平均损失: {np.mean(losses[-10:]):.4f}")
            print()
    
    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), '/mnt/user-data/outputs/final_model.pth')
    
    # 绘制训练曲线
    plot_training_results(episode_rewards, episode_lengths, losses)
    
    return agent, env


def plot_training_results(rewards, lengths, losses):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(rewards, alpha=0.3)
    axes[0, 0].plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    axes[0, 0].set_title('Episode Rewards', fontsize=14)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # 步数曲线
    axes[0, 1].plot(lengths, alpha=0.3)
    axes[0, 1].plot(np.convolve(lengths, np.ones(50)/50, mode='valid'), linewidth=2)
    axes[0, 1].set_title('Episode Lengths', fontsize=14)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # 损失曲线
    if len(losses) > 0:
        axes[1, 0].plot(losses, alpha=0.3)
        axes[1, 0].plot(np.convolve(losses, np.ones(50)/50, mode='valid'), linewidth=2)
        axes[1, 0].set_title('Training Loss', fontsize=14)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # 成功率曲线
    window = 50
    success = [1 if r > 0 else 0 for r in rewards]
    success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
    axes[1, 1].plot(success_rate, linewidth=2)
    axes[1, 1].set_title(f'Success Rate (window={window})', fontsize=14)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_results.png', dpi=150)
    print("训练结果已保存到: /mnt/user-data/outputs/training_results.png")
    

def visualize_path(env, agent, num_steps=100):
    """可视化路径规划"""
    state = env.reset()
    
    # 记录路径
    positions = [env.position.copy()]
    aps_used = [env.current_ap_idx]
    
    done = False
    step = 0
    while not done and step < num_steps:
        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
        
        positions.append(env.position.copy())
        aps_used.append(env.current_ap_idx)
        step += 1
    
    # 绘制
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 绘制所有AP
    for i, ap in enumerate(env.aps):
        if i == env.start_ap_idx:
            color = 'green'
            size = 200
            marker = '^'
            label = 'Start AP'
        elif i == env.target_ap_idx:
            color = 'red'
            size = 200
            marker = 's'
            label = 'Target AP'
        else:
            color = 'blue'
            size = 100
            marker = 's'
            label = None
        
        ax.scatter(ap[0], ap[1], c=color, s=size, alpha=0.6, marker=marker, 
                  edgecolors='black', label=label if label and i <= 1 else None)
        ax.text(ap[0], ap[1]+20, ap[2], fontsize=7, ha='center')
    
    # 绘制STA路径
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=2, label='STA Path', alpha=0.7)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], c='orange', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='End', zorder=5)
    
    # 绘制使用的AP连接
    for i in range(len(aps_used) - 1):
        ap_idx = aps_used[i]
        ap_pos = env.aps[ap_idx][:2]
        sta_pos = positions[i]
        ax.plot([sta_pos[0], ap_pos[0]], [sta_pos[1], ap_pos[1]], 
                'c--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Mine Communication Path Planning using DQN\n(Blue: APs, Red: Target AP, Green: STA Path)', 
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/path_visualization.png', dpi=150)
    print("路径可视化已保存到: /mnt/user-data/outputs/path_visualization.png")
    print(f"\n路径信息:")
    print(f"  总步数: {len(positions)}")
    print(f"  AP切换次数: {len(set(aps_used)) - 1}")
    print(f"  最终距离目标: {np.linalg.norm(positions[-1] - env.target_position):.1f}m")


if __name__ == "__main__":
    print("="*60)
    print("矿道WiFi-HaLow通信路径规划 - 基于DQN深度强化学习")
    print("="*60)
    
    # ==================== 分支配置 ====================
    # 您可以在这里自定义分支矿道的数量和位置
    # 每个分支配置包含：
    #   'x': 分支点的x坐标（米）
    #   'has_up': 是否有上分支（向y正方向）
    #   'has_down': 是否有下分支（向y负方向）
    #   'length': 分支长度（米）
    
    # 示例1：默认配置（2个分支点，各有上下分支）
    # BRANCH_CONFIG = [
    #     {'x': 200, 'has_up': True, 'has_down': True, 'length': 600},
    #     {'x': 700, 'has_up': True, 'has_down': True, 'length': 600}
    # ]
    
    # 示例2：更多分支（4个分支点）
    BRANCH_CONFIG = [
        {'x': 200, 'has_up': True, 'has_down': True, 'length': 600},
        {'x': 400, 'has_up': True, 'has_down': False, 'length': 400},  # 只有上分支
        {'x': 700, 'has_up': True, 'has_down': True, 'length': 600},
        {'x': 1000, 'has_up': False, 'has_down': True, 'length': 500}  # 只有下分支
    ]
    
    # 示例3：更密集的分支布局（6个分支点）
    # BRANCH_CONFIG = [
    #     {'x': 150, 'has_up': True, 'has_down': True, 'length': 400},
    #     {'x': 350, 'has_up': True, 'has_down': True, 'length': 500},
    #     {'x': 550, 'has_up': True, 'has_down': False, 'length': 300},
    #     {'x': 750, 'has_up': False, 'has_down': True, 'length': 450},
    #     {'x': 950, 'has_up': True, 'has_down': True, 'length': 600},
    #     {'x': 1150, 'has_up': True, 'has_down': True, 'length': 350}
    # ]
    
    # 示例4：简化配置（只有1个分支点）
    # BRANCH_CONFIG = [
    #     {'x': 600, 'has_up': True, 'has_down': True, 'length': 800}
    # ]
    
    # 示例5：不对称分支
    # BRANCH_CONFIG = [
    #     {'x': 200, 'has_up': True, 'has_down': False, 'length': 800},   # 只有上分支，较长
    #     {'x': 600, 'has_up': False, 'has_down': True, 'length': 400},   # 只有下分支，较短
    #     {'x': 1000, 'has_up': True, 'has_down': True, 'length': 600}    # 上下都有
    # ]
    # ==================================================
    
    # 首先创建一个环境来显示所有AP位置
    print("\n正在根据配置创建矿道环境...")
    print("-" * 60)
    temp_env = MineEnvironment(branch_config=BRANCH_CONFIG)
    print("-" * 60)
    
    print("\n可用的AP列表:")
    print("-" * 60)
    for i, ap in enumerate(temp_env.aps):
        print(f"  索引 {i:2d}: {ap[2]:25s} at ({ap[0]:6.1f}, {ap[1]:6.1f})")
    print("-" * 60)
    
    # 训练参数
    # 您可以修改这些参数来改变起始位置和目标位置
    START_AP_IDX = 0      # 起始AP索引（0=最左侧第一个AP）
    TARGET_AP_IDX = 68  # 目标AP索引（None=自动选择最右侧主矿道AP）
    NUM_EPISODES = 500    # 训练回合数
    
    print(f"\n训练配置:")
    print(f"  起始AP索引: {START_AP_IDX}")
    print(f"  目标AP索引: {TARGET_AP_IDX if TARGET_AP_IDX is not None else '自动(最右侧)'}")
    print(f"  训练回合数: {NUM_EPISODES}")
    print(f"  分支数量: {len(BRANCH_CONFIG)}")
    
    # 训练
    print("\n开始训练...")
    agent, env = train_dqn(
        num_episodes=NUM_EPISODES, 
        start_ap_idx=START_AP_IDX, 
        target_ap_idx=TARGET_AP_IDX,
        branch_config=BRANCH_CONFIG
    )
    
    # 可视化最优路径
    print("\n生成最优路径可视化...")
    visualize_path(env, agent)
    
    print("\n训练完成！")
    print(f"模型已保存到: /mnt/user-data/outputs/")
    print("\n提示: 要修改分支配置，请编辑代码中的 BRANCH_CONFIG 参数")
    print("      要修改起始位置，请编辑代码中的 START_AP_IDX 参数")
    print("      AP索引可以从上面的AP列表中查看")
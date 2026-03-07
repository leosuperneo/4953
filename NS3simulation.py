# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:18:23 2025

@author: HONGTAO LEO
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE 802.11ah 信号强度热力图可视化工具
完整的Python实现

使用方法:
1. 安装依赖: pip install matplotlib numpy scipy
2. 运行脚本: python heatmap_visualizer.py
3. 可选: python heatmap_visualizer.py --input visualization_data.json

作者: NS-3 仿真分析工具
日期: 2024
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import argparse
from pathlib import Path

# 默认的32个STA完整数据
DEFAULT_DATA = {
    "apData": [
        {"id": 0, "x": 0, "y": 0},
        {"id": 1, "x": 500, "y": 0},
        {"id": 2, "x": 1000, "y": 0}
    ],
    "staData": [
        {"id": 0, "x": 45.23, "y": 30.15, "rssi": -58.23, "snr": 32.45, "distance": 45.23, "packetLoss": 0.00, "goodput": 7.68},
        {"id": 1, "x": 52.67, "y": -25.30, "rssi": -59.12, "snr": 31.87, "distance": 52.67, "packetLoss": 0.00, "goodput": 7.71},
        {"id": 2, "x": 68.91, "y": 15.45, "rssi": -60.45, "snr": 30.56, "distance": 68.91, "packetLoss": 0.00, "goodput": 7.65},
        {"id": 3, "x": 38.45, "y": -10.20, "rssi": -57.89, "snr": 33.12, "distance": 38.45, "packetLoss": 0.00, "goodput": 7.69},
        {"id": 4, "x": 89.23, "y": 20.50, "rssi": -62.34, "snr": 28.67, "distance": 89.23, "packetLoss": 16.67, "goodput": 6.42},
        {"id": 5, "x": 48.90, "y": 35.20, "rssi": -58.67, "snr": 32.34, "distance": 48.90, "packetLoss": 0.00, "goodput": 7.72},
        {"id": 6, "x": 95.67, "y": -40.15, "rssi": -63.12, "snr": 27.89, "distance": 95.67, "packetLoss": 16.67, "goodput": 6.38},
        {"id": 7, "x": 61.23, "y": 25.80, "rssi": -59.45, "snr": 31.56, "distance": 61.23, "packetLoss": 0.00, "goodput": 7.66},
        {"id": 8, "x": 82.45, "y": -30.60, "rssi": -61.78, "snr": 29.23, "distance": 82.45, "packetLoss": 16.67, "goodput": 6.45},
        {"id": 9, "x": 51.34, "y": 12.40, "rssi": -58.90, "snr": 32.11, "distance": 51.34, "packetLoss": 0.00, "goodput": 7.70},
        {"id": 10, "x": 65.78, "y": -20.90, "rssi": -59.78, "snr": 31.23, "distance": 65.78, "packetLoss": 0.00, "goodput": 7.67},
        {"id": 11, "x": 142.34, "y": 35.60, "rssi": -68.45, "snr": 21.56, "distance": 142.34, "packetLoss": 33.33, "goodput": 5.12},
        {"id": 12, "x": 42.11, "y": 5.30, "rssi": -57.34, "snr": 33.67, "distance": 42.11, "packetLoss": 0.00, "goodput": 7.73},
        {"id": 13, "x": 91.56, "y": -15.70, "rssi": -62.67, "snr": 28.34, "distance": 91.56, "packetLoss": 16.67, "goodput": 6.41},
        {"id": 14, "x": 49.67, "y": 18.90, "rssi": -58.56, "snr": 32.45, "distance": 49.67, "packetLoss": 0.00, "goodput": 7.68},
        {"id": 15, "x": 98.23, "y": 45.20, "rssi": -63.45, "snr": 27.56, "distance": 98.23, "packetLoss": 16.67, "goodput": 6.39},
        {"id": 16, "x": 558.90, "y": 30.40, "rssi": -59.23, "snr": 31.78, "distance": 58.90, "packetLoss": 0.00, "goodput": 7.69},
        {"id": 17, "x": 546.78, "y": -20.50, "rssi": -58.12, "snr": 32.89, "distance": 46.78, "packetLoss": 0.00, "goodput": 7.71},
        {"id": 18, "x": 638.67, "y": 40.80, "rssi": -67.89, "snr": 22.12, "distance": 138.67, "packetLoss": 33.33, "goodput": 5.15},
        {"id": 19, "x": 570.45, "y": 25.60, "rssi": -60.12, "snr": 30.89, "distance": 70.45, "packetLoss": 0.00, "goodput": 7.65},
        {"id": 20, "x": 587.89, "y": -15.30, "rssi": -62.01, "snr": 28.99, "distance": 87.89, "packetLoss": 16.67, "goodput": 6.43},
        {"id": 21, "x": 544.56, "y": 18.20, "rssi": -57.67, "snr": 33.34, "distance": 44.56, "packetLoss": 0.00, "goodput": 7.72},
        {"id": 22, "x": 601.23, "y": 50.40, "rssi": -63.78, "snr": 27.23, "distance": 101.23, "packetLoss": 16.67, "goodput": 6.37},
        {"id": 23, "x": 559.34, "y": 22.70, "rssi": -59.01, "snr": 31.99, "distance": 59.34, "packetLoss": 0.00, "goodput": 7.70},
        {"id": 24, "x": 550.12, "y": -25.80, "rssi": -58.45, "snr": 32.56, "distance": 50.12, "packetLoss": 0.00, "goodput": 7.68},
        {"id": 25, "x": 645.89, "y": -40.30, "rssi": -69.12, "snr": 20.89, "distance": 145.89, "packetLoss": 33.33, "goodput": 5.18},
        {"id": 26, "x": 1072.34, "y": 18.50, "rssi": -60.34, "snr": 30.67, "distance": 72.34, "packetLoss": 0.00, "goodput": 7.66},
        {"id": 27, "x": 1085.67, "y": -10.20, "rssi": -61.45, "snr": 29.56, "distance": 85.67, "packetLoss": 16.67, "goodput": 6.44},
        {"id": 28, "x": 1040.23, "y": 8.60, "rssi": -57.12, "snr": 33.89, "distance": 40.23, "packetLoss": 0.00, "goodput": 7.73},
        {"id": 29, "x": 1103.45, "y": 25.80, "rssi": -64.12, "snr": 26.89, "distance": 103.45, "packetLoss": 16.67, "goodput": 6.40},
        {"id": 30, "x": 1047.89, "y": -18.30, "rssi": -58.34, "snr": 32.67, "distance": 47.89, "packetLoss": 0.00, "goodput": 7.71},
        {"id": 31, "x": 1141.67, "y": -38.90, "rssi": -68.78, "snr": 21.23, "distance": 141.67, "packetLoss": 33.33, "goodput": 5.20}
    ],
    "config": {
        "simulationTime": 60,
        "numAPs": 3,
        "numSTAs": 32,
        "dataMode": "MCS2_0",
        "bandwidth": 2,
        "datarate": 7.8,
        "payloadSize": 100,
        "trafficInterval": 10000
    }
}


class SignalHeatmapVisualizer:
    """IEEE 802.11ah 信号强度热力图可视化器"""
    
    def __init__(self, data=None):
        """初始化可视化器
        
        Args:
            data: 包含apData, staData, config的字典，如果为None则使用默认数据
        """
        if data is None:
            data = DEFAULT_DATA
            
        self.ap_data = data.get('apData', [])
        self.sta_data = data.get('staData', [])
        self.config = data.get('config', {})
        
        # 设置图形风格
        plt.style.use('dark_background')
        self.fig = None
        self.ax = None
        
    def calculate_signal_at_point(self, x, y):
        """计算指定点的理论信号强度（使用路径损耗模型）
        
        Args:
            x, y: 坐标点
            
        Returns:
            float: RSSI值 (dBm)
        """
        max_signal = -100
        for ap in self.ap_data:
            distance = np.sqrt((x - ap['x'])**2 + (y - ap['y'])**2)
            # 简化的路径损耗模型: RSSI = -30 - 37*log10(d)
            rssi = -30 - 37 * np.log10(max(1, distance))
            max_signal = max(max_signal, rssi)
        return max_signal
    
    def get_rssi_color(self, rssi):
        """根据RSSI值获取颜色
        
        Args:
            rssi: RSSI值 (dBm)
            
        Returns:
            tuple: RGB颜色值 (0-1范围)
        """
        # 归一化: -40(强) 到 -80(弱)
        normalized = max(0, min(1, (-rssi - 40) / 40))
        
        if normalized < 0.25:
            return (0, 1, 0)  # 绿色
        elif normalized < 0.5:
            return (0.5, 1, 0)  # 黄绿
        elif normalized < 0.75:
            return (1, 1, 0)  # 黄色
        else:
            return (1, 1 - normalized, 0)  # 橙红
    
    def plot_heatmap(self, metric='rssi', show_labels=True, 
                     show_heatmap=True, resolution=25,
                     show_poor_only=False, output_file=None):
        """绘制热力图
        
        Args:
            metric: 'rssi' 或 'snr'
            show_labels: 是否显示标签
            show_heatmap: 是否显示背景热力图
            resolution: 热力图分辨率(米)
            show_poor_only: 是否只显示问题STA
            output_file: 输出文件路径，如果为None则显示图形
        """
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.ax.set_facecolor('#1a1a1a')
        self.fig.patch.set_facecolor('#2d2d2d')
        
        # 计算绘图范围
        all_x = [ap['x'] for ap in self.ap_data] + [sta['x'] for sta in self.sta_data]
        all_y = [ap['y'] for ap in self.ap_data] + [sta['y'] for sta in self.sta_data]
        
        x_min, x_max = min(all_x) - 100, max(all_x) + 100
        y_min, y_max = min(all_y) - 100, max(all_y) + 100
        
        # 绘制背景热力图
        if show_heatmap:
            self._draw_background_heatmap(x_min, x_max, y_min, y_max, resolution)
        
        # 绘制AP覆盖范围圆
        self._draw_ap_coverage()
        
        # 过滤STA
        displayed_stas = self.sta_data
        if show_poor_only:
            displayed_stas = [sta for sta in self.sta_data if sta.get('packetLoss', 0) > 10]
        
        # 绘制STA
        self._draw_stas(displayed_stas, metric, show_labels)
        
        # 绘制AP
        self._draw_aps()
        
        # 设置坐标轴
        self.ax.set_xlabel('distance (m)', fontsize=12, color='white')
        self.ax.set_ylabel('distance (m)', fontsize=12, color='white')
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.set_aspect('equal')
        
        # 添加标题和统计信息
        self._add_title_and_stats(metric, show_poor_only)
        
        # 添加图例
        self._add_legend(metric)
        
        plt.tight_layout()
        
        # 保存或显示
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#2d2d2d')
            print(f"✅ 图像已保存到: {output_file}")
        else:
            plt.show()
    
    def _draw_background_heatmap(self, x_min, x_max, y_min, y_max, resolution):
        """绘制背景热力图"""
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 计算每个网格点的信号强度
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.calculate_signal_at_point(X[i, j], Y[i, j])
        
        # 创建自定义颜色映射
        colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('signal', colors, N=n_bins)
        
        # 绘制热力图
        im = self.ax.pcolormesh(X, Y, Z, cmap=cmap, alpha=0.4, 
                                vmin=-80, vmax=-40, shading='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=self.ax, pad=0.02)
        cbar.set_label('信号强度 (dBm)', fontsize=10, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
    
    def _draw_ap_coverage(self):
        """绘制AP覆盖范围圆"""
        for ap in self.ap_data:
            for radius in [50, 100, 150, 200]:
                circle = Circle((ap['x'], ap['y']), radius, 
                              fill=False, edgecolor='cyan', 
                              alpha=0.3 - radius/1000, linewidth=1)
                self.ax.add_patch(circle)
    
    def _draw_stas(self, stas, metric, show_labels):
        """绘制STA节点"""
        for sta in stas:
            value = sta.get(metric, sta.get('rssi', -70))
            color = self.get_rssi_color(value if metric == 'rssi' else 
                                       40 + value - 30)  # SNR转换
            
            # 绘制STA圆点
            self.ax.plot(sta['x'], sta['y'], 'o', 
                        color=color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2,
                        zorder=10)
            
            # 添加标签
            if show_labels:
                label = f"STA{sta['id']}\n{value:.1f}"
                if metric == 'rssi':
                    label += "dBm"
                else:
                    label += "dB"
                    
                self.ax.text(sta['x'] + 8, sta['y'], label,
                           fontsize=7, color='white',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', alpha=0.7),
                           zorder=11)
                
                # 显示丢包率
                if sta.get('packetLoss', 0) > 0:
                    self.ax.text(sta['x'] + 8, sta['y'] - 15,
                               f"Loss:{sta['packetLoss']:.0f}%",
                               fontsize=6, color='red',
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='black', alpha=0.7),
                               zorder=11)
    
    def _draw_aps(self):
        """绘制AP节点"""
        for ap in self.ap_data:
            # AP三角形
            triangle = Polygon([
                [ap['x'], ap['y'] + 15],
                [ap['x'] - 12, ap['y'] - 8],
                [ap['x'] + 12, ap['y'] - 8]
            ], facecolor='red', edgecolor='white', linewidth=2, zorder=20)
            self.ax.add_patch(triangle)
            
            # 发射波纹
            for radius in [10, 20, 30]:
                circle = Circle((ap['x'], ap['y']), radius,
                              fill=False, edgecolor='red',
                              alpha=0.6 - radius/50, linewidth=2, zorder=19)
                self.ax.add_patch(circle)
            
            # AP标签
            self.ax.text(ap['x'], ap['y'] - 30, f"AP {ap['id']}",
                        fontsize=12, fontweight='bold', color='white',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='red', alpha=0.8),
                        zorder=21)
            
            self.ax.text(ap['x'], ap['y'] - 45, f"({ap['x']}, {ap['y']})",
                        fontsize=9, color='lightgray', ha='center',
                        zorder=21)
    
    def _add_title_and_stats(self, metric, show_poor_only):
        """添加标题和统计信息"""
        metric_name = "RSSI (signal strength)" if metric == 'rssi' else "SNR"
        
        # 计算统计数据
        values = [sta.get(metric, sta.get('rssi', -70)) for sta in self.sta_data]
        avg_value = np.mean(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        packet_losses = [sta.get('packetLoss', 0) for sta in self.sta_data]
        avg_loss = np.mean(packet_losses)
        poor_stas = sum(1 for loss in packet_losses if loss > 10)
        excellent_stas = sum(1 for loss in packet_losses if loss == 0)
        
        title = f"IEEE 802.11ah Signal strength heat map - {metric_name}\n"
        title += f"AP numb: {len(self.ap_data)} | STA numb: {len(self.sta_data)} | "
        title += f"Good STA: {excellent_stas} | Bad STA: {poor_stas}\n"
        
        if metric == 'rssi':
            title += f"Mean RSSI: {avg_value:.2f} dBm | Range: [{min_value:.1f}, {max_value:.1f}] dBm | "
        else:
            title += f"mean SNR: {avg_value:.2f} dB | Range: [{min_value:.1f}, {max_value:.1f}] dB | "
        
        title += f"mean loss: {avg_loss:.2f}%"
        
        if show_poor_only:
            title += "\n⚠️ 仅显示问题STA (丢包率 > 10%)"
        
        self.ax.set_title(title, fontsize=13, fontweight='bold', 
                         color='white', pad=20)
    
    def _add_legend(self, metric):
        """添加图例"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                  markersize=12, label='AP', markeredgecolor='white', 
                  markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=10, label='STA(strong)', markeredgecolor='white',
                  markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                  markersize=10, label='STA(medium)', markeredgecolor='white',
                  markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                  markersize=10, label='STA(weak)', markeredgecolor='white',
                  markeredgewidth=2),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right',
                      fontsize=10, framealpha=0.9, facecolor='#2d2d2d',
                      edgecolor='white')
    
    def print_statistics(self):
        """打印详细统计信息"""
        print("\n" + "="*80)
        print("📊 网络性能统计报告")
        print("="*80)
        
        # 基本信息
        print(f"\n【配置信息】")
        print(f"  AP数量: {len(self.ap_data)}")
        print(f"  STA数量: {len(self.sta_data)}")
        if self.config:
            print(f"  仿真时间: {self.config.get('simulationTime', 'N/A')}秒")
            print(f"  数据模式: {self.config.get('dataMode', 'N/A')}")
            print(f"  带宽: {self.config.get('bandwidth', 'N/A')} MHz")
        
        # RSSI统计
        rssi_values = [sta['rssi'] for sta in self.sta_data]
        print(f"\n【RSSI统计】")
        print(f"  平均: {np.mean(rssi_values):.2f} dBm")
        print(f"  最小: {np.min(rssi_values):.2f} dBm")
        print(f"  最大: {np.max(rssi_values):.2f} dBm")
        print(f"  标准差: {np.std(rssi_values):.2f} dBm")
        
        # SNR统计
        snr_values = [sta['snr'] for sta in self.sta_data]
        print(f"\n【SNR统计】")
        print(f"  平均: {np.mean(snr_values):.2f} dB")
        print(f"  最小: {np.min(snr_values):.2f} dB")
        print(f"  最大: {np.max(snr_values):.2f} dB")
        
        # 丢包统计
        packet_losses = [sta.get('packetLoss', 0) for sta in self.sta_data]
        excellent_stas = [sta for sta in self.sta_data if sta.get('packetLoss', 0) == 0]
        poor_stas = [sta for sta in self.sta_data if sta.get('packetLoss', 0) > 10]
        
        print(f"\n【丢包统计】")
        print(f"  平均丢包率: {np.mean(packet_losses):.2f}%")
        print(f"  优秀STA (0%丢包): {len(excellent_stas)}个")
        print(f"  问题STA (>10%丢包): {len(poor_stas)}个")
        
        if poor_stas:
            print(f"\n【问题STA列表】")
            poor_stas_sorted = sorted(poor_stas, 
                                     key=lambda x: x.get('packetLoss', 0), 
                                     reverse=True)
            for sta in poor_stas_sorted[:10]:  # 只显示前10个
                print(f"  STA {sta['id']:2d}: "
                      f"距离={sta['distance']:6.1f}m, "
                      f"RSSI={sta['rssi']:6.2f}dBm, "
                      f"丢包={sta.get('packetLoss', 0):5.1f}%")
        
        # 距离分析
        print(f"\n【按距离分段分析】")
        distance_ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
        for min_d, max_d in distance_ranges:
            stas_in_range = [sta for sta in self.sta_data 
                           if min_d <= sta['distance'] < max_d]
            if stas_in_range:
                avg_loss = np.mean([sta.get('packetLoss', 0) 
                                  for sta in stas_in_range])
                avg_rssi = np.mean([sta['rssi'] for sta in stas_in_range])
                print(f"  {min_d:3d}-{max_d:3d}m: "
                      f"{len(stas_in_range):2d}个STA, "
                      f"平均丢包={avg_loss:5.2f}%, "
                      f"平均RSSI={avg_rssi:.2f}dBm")
        
        print("\n" + "="*80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='IEEE 802.11ah 信号强度热力图可视化工具'
    )
    parser.add_argument('-i', '--input', type=str, default=None,
                       help='输入JSON数据文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出图像文件路径(png/pdf/svg)')
    parser.add_argument('-m', '--metric', type=str, default='rssi',
                       choices=['rssi', 'snr'],
                       help='显示指标: rssi 或 snr (默认: rssi)')
    parser.add_argument('--no-labels', action='store_true',
                       help='不显示标签')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='不显示背景热力图')
    parser.add_argument('-r', '--resolution', type=int, default=25,
                       help='热力图分辨率(米) (默认: 25)')
    parser.add_argument('--poor-only', action='store_true',
                       help='只显示问题STA')
    parser.add_argument('--stats', action='store_true',
                       help='打印详细统计信息')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载数据文件: {args.input}")
        except Exception as e:
            print(f"❌ 加载数据文件失败: {e}")
            print("使用默认数据...")
            data = None
    else:
        print("使用默认的32个STA数据")
        data = None
    
    # 创建可视化器
    visualizer = SignalHeatmapVisualizer(data)
    
    # 打印统计信息
    if args.stats or not args.output:
        visualizer.print_statistics()
    
    # 绘制热力图
    visualizer.plot_heatmap(
        metric=args.metric,
        show_labels=not args.no_labels,
        show_heatmap=not args.no_heatmap,
        resolution=args.resolution,
        show_poor_only=args.poor_only,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
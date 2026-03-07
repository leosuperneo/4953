% ======== Global style =========
close all; clear; clc;
set(0,'DefaultAxesFontName','Arial');
set(0,'DefaultAxesFontSize',12);
set(0,'DefaultLineLineWidth',2);
set(0,'DefaultAxesGridAlpha',0.15);

% 固定颜色（色盲友好系）
c1 = [0.00 0.45 0.74]; % LoS
c2 = [0.85 0.33 0.10]; % NLoS
c3 = [0.47 0.67 0.19]; % Wall

fig = figure('Color','w','Position',[100 80 1400 900]);
tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
sgtitle(tl,'Experiment Design','FontSize',18,'FontWeight','bold');

% ======== 左上：竖切面结构示意（占左列上半）========
ax1 = nexttile(tl,1,[1 1]); % 左上
hold(ax1,'on'); axis(ax1,'equal');
% 这里替换成你的截面绘图函数/数据：
% demo：三层走廊占位图
h=3; L=120; w=3;
for k=0:2
    rectangle('Position',[0, k*(h+0.5), L, h],'EdgeColor',[0.6 0.6 0.6],'LineWidth',1.2);
    text(2, k*(h+0.5)+h/2, sprintf('Level %d | Clear height = %.2fm',3-k,h), ...
        'VerticalAlignment','middle','FontSize',12);
end
xlim([0 L]); ylim([0 3*(h+0.5)]); 
xlabel('Corridor length, L (m)'); ylabel('Elevation (m)');
title('Vertical Cross-Section');

% ======== 左下：RSS 理论 vs 实测（占左列下半）========
ax2 = nexttile(tl,4,[1 1]); % 左下
hold(ax2,'on'); grid on;
% 你的数据：d, rss_theory, rss_true
% ——下面用占位数据演示——
d = (1:100)'; 
rss_theory = -40 - 10*2*log10(d);
rss_true = rss_theory + (randn(size(d))*6 - 3); % demo 噪声
plot(d, rss_theory,'Color',c1,'DisplayName','Theory RSS');
scatter(d, rss_true,16,'filled','MarkerFaceAlpha',.6,'MarkerEdgeColor','none','DisplayName','Measured RSS');
xlabel('Distance d (m)'); ylabel('RSS (dBm)');
title('RSS vs Distance (Theory vs Measured)');
legend('Location','southoutside','Orientation','horizontal','Box','off');
xlim([0 100]); ylim([-105 -30]);

% ======== 右侧四图：性能曲线（2x2）========
% demo 数据（请替换为你的 LoS/NLoS/Wall 三组）
x = 10:10:90;
rss_los  = -35 - 0.25*(x-10);
rss_nlos = -55 - 0.9*(x-10);
rss_wall = -60 - 0.4*(x-10);

snr_los  = 45 - 0.3*(x-10);
snr_nlos = 30 - 0.5*(x-10);
snr_wall = 28 - 0.15*(x-10);

rtt_los  = 5 + 0.02*(x-10);
rtt_nlos = 20 + 1.3*(x-10);
rtt_wall = 35 + 0.25*(x-10);

per_los  = max(0, 0.001*(x-70).^2);
per_nlos = min(100, 5 + 1.9*(x-10));
per_wall = 35 + 0.35*(x-10);

% 右上：RSSI
ax3 = nexttile(tl,2); hold(ax3,'on'); grid on;
plot(x, rss_los ,'o-','Color',c1,'DisplayName','LoS','MarkerSize',5);
plot(x, rss_nlos,'s-','Color',c2,'DisplayName','NLoS','MarkerSize',5);
plot(x, rss_wall,'^-','Color',c3,'DisplayName','Wall','MarkerSize',5);
xlabel('Distance (m)'); ylabel('RSSI (dBm)'); title('RSSI');
legend('Location','southoutside','Orientation','horizontal','Box','off'); 
ylim([-95 -30]);

% 右上中：SNR
ax4 = nexttile(tl,3); hold(ax4,'on'); grid on;
plot(x, snr_los ,'o-','Color',c1,'DisplayName','LoS','MarkerSize',5);
plot(x, snr_nlos,'s-','Color',c2,'DisplayName','NLoS','MarkerSize',5);
plot(x, snr_wall,'^-','Color',c3,'DisplayName','Wall','MarkerSize',5);
xlabel('Distance (m)'); ylabel('SNR (dB)'); title('SNR'); ylim([10 50]);

% 右下左：RTT
ax5 = nexttile(tl,5); hold(ax5,'on'); grid on;
plot(x, rtt_los ,'o-','Color',c1,'MarkerSize',5,'DisplayName','LoS');
plot(x, rtt_nlos,'s-','Color',c2,'MarkerSize',5,'DisplayName','NLoS');
plot(x, rtt_wall,'^-','Color',c3,'MarkerSize',5,'DisplayName','Wall');
xlabel('Distance (m)'); ylabel('RTT (ms)'); title('RTT');

% 右下右：PER
ax6 = nexttile(tl,6); hold(ax6,'on'); grid on;
plot(x, per_los ,'o-','Color',c1,'MarkerSize',5,'DisplayName','LoS');
plot(x, per_nlos,'s-','Color',c2,'MarkerSize',5,'DisplayName','NLoS');
plot(x, per_wall,'^-','Color',c3,'MarkerSize',5,'DisplayName','Wall');
xlabel('Distance (m)'); ylabel('PER (%)'); title('PER'); ylim([0 100]);

% 统一 x 轴范围
set([ax3 ax4 ax5 ax6],'XLim',[10 90]);

% ======== 导出高分辨率 ========
exportgraphics(fig,'experiment_design.pdf','ContentType','vector','Resolution',300);
exportgraphics(fig,'experiment_design.png','Resolution',300);

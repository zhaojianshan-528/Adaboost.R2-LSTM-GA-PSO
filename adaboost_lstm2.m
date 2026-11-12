%% ================== 多输出 LSTM + 窗口平移，滚动预测 100 步（归一化修正版） ==================
clear; close all; clc;

%% 读数与基本设置
raw = readmatrix('data.csv');            % 原始文件
data = raw(:, 2:10);% 使用 9 列：[8 外生 + 1 负荷]
data(:,1)=data(:,1)*9;
idxTarget = 1;                           % 负荷在 data 中的列索引(若不是第1列请修改)

w = 1;                                   % 步距
s = 24;                                  % 窗口长度（小时）
m = 1200;                                % 训练样本数
n = 200;                                 % 测试样本数                               % 滚动预测步数（小时）

[numRows, numCols] = size(data);         % numCols 应为 9

%% 组装训练/测试滑动窗口输入（展平 24*9=216 维）
input_train = [];
for i = 1:m
    xx = data(1 + w*(i-1) : w*(i-1) + s, :);  % [24 x 9]
    input_train = [input_train, xx(:)];       % → [216 x m]
end

% 训练输出：下一小时的“全部 9 维”（多输出）
output_train_all = data(2:m+1, :)';           % [9 x m]
output_train=data(1:m,1);
% 测试集输入窗口（用于 one-step 对齐评估 & 取 seed）
input_test = [];
for i = m+1 : m+n
    xx = data(1 + w*(i-1) : w*(i-1) + s, :);
    input_test = [input_test, xx(:)];         % [216 x n]
end

% 测试集对齐真值（下一小时）
output_test_all = data(m+2 : m+n+1, :)';      % [9 x n]
output_test_y   = output_test_all(idxTarget,:);

%% ================== 归一化（只用训练集拟合缩放器！） ==================
[inputn,  inputps]  = mapminmax(input_train, 0, 1);        % [216 x m]
[outputn, outputps] = mapminmax(output_train_all, 0, 1);   % [  9 x m]
inputn_test         = mapminmax('apply', input_test, inputps);
[output_testn,output_testps]=mapminmax(output_test_all,0,1);
%% ================== 网络结构（多输出：9 维） ==================
inputnum   = size(inputn, 1);   % 216
outputnum  = size(outputn, 1);  % 9
numHidden  = 64;

layers = [
    sequenceInputLayer(inputnum)                 % 216
    lstmLayer(numHidden, 'OutputMode', 'sequence')
    dropoutLayer(0.1,'Name','drop1')  
    fullyConnectedLayer(outputnum)               % 9
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 600, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% ================== 训练 ==================
net0 = trainNetwork(inputn, outputn, layers, options);
%% ================== 单步测试（对齐评估） ==================
an0_n = predict(net0, inputn_test);                         % [9 x n]
test_simu_all = mapminmax('reverse', an0_n, outputps);      % 反归一化 → [9 x n]
test_simu_y   = test_simu_all(idxTarget,:);                 % 取负荷
mse0 = mse(output_test_y, test_simu_y);

%% ================== 真实未来（如数据足够） ==================
% 未来 H 步的真值（从 m+n+1 开始）
                         % 防越界
            % 长度可能 < H

%% ================== 可视化 ==================
% 1) 单步预测对齐评估
figure;
plot(output_test_y, 'b-','LineWidth',1.2); hold on;
plot(test_simu_y, 'r--','LineWidth',1.2);
title(['One-step MSE = ', num2str(mse0)]);
legend('True (test, next-hour y)', 'One-step LSTM');
xlabel('Sample Index'); ylabel('Load'); grid on;

% % 2) 滚动 100 步预测（与真值对比，如可用）
% figure;
% plot(1:H, yhat_roll, 'm-','LineWidth',1.6); hold on;
% if ~isempty(truth_future)
%     plot(1:numel(truth_future), truth_future, 'b-','LineWidth',1.4);
%     legend('Rolled forecast (y)', 'True future (y)');
% else
%     legend('Rolled forecast (y)');
% end
% title('Rolling Forecast (H=100) from the last known window');
% xlabel('Horizon (hour)'); ylabel('Load'); grid on;
% s=1200;
% % 3) 最近 24h 与未来 H 步（可视化衔接感）
% hist_tail = data(1:s, :);                 % [24 x 9]
% hist_load = hist_tail(:, idxTarget)';             % [1 x 24]
% figure;
% plot(1:s,        hist_load, 'k-','LineWidth',1.2); hold on;
% plot(s+(1:H),    yhat_roll, 'g-','LineWidth',1.6);
% legend('Recent 24h (true y)', 'Next 100h (rolled y)');
% xlabel('Time (hour, relative)'); ylabel('Load');
% title('Recent vs. Rolled Forecast'); grid on;
% 
% disp('/////////////////////////////////');
% disp('已用一致的归一化缩放器完成热启动与滚动预测。');


%% 
%% —— AdaBoost.R2 (MSE-oriented) + LSTM —— 
%% —— AdaBoost.R2 (MSE-oriented) + LSTM ——  (改进版，聚焦目标维 & 正统权重更新)
disp('AdaBoost.R2 (MSE-oriented) + LSTM  — 改进版开始');

K = 9;                                   % 弱学习器数量（可调）
loss_power = 2;                           % 与 MSE 对齐：L = (|e|/R)^p
[mm, nn] = size(inputn);                  % inputn: [216 x m]
D = ones(1, nn) / nn;                     % 训练样本权重分布

% 仅针对“目标维”做缩放（弱学习器只输出 1 维）
% 你前面已有 output_train_all; 目标真值(原尺度)为 output_train = data(1:m,1)
y_train_true  = data(1:nn, 1)';           % [1 x m] 注意：此处 nn == m
[yn_train, yps_train] = mapminmax(y_train_true, 0, 1);   % [1 x m]
y_test_true   = output_test_y(:)';        % [1 x n]（已由你前文得到）

alpha = zeros(1, K);
test_preds = zeros(K, size(inputn_test, 2));  % 每轮在测试集上的预测（原尺度）
train_preds_all = zeros(K, nn);               % 每轮在训练集上的预测（原尺度）

% 验证集索引（用于 stacking，不参与权重更新）
n_val  = max(1, round(0.1 * nn));
val_idx = nn - n_val + 1 : nn;
trn_idx = 1 : (nn - n_val);
has_val = numel(val_idx) >= 5;

for t = 1:K
    fprintf('第 %d 个弱学习器（按 D 加权自助采样）...\n', t);

    % === 按当前分布 D 自助采样（bootstrap with replacement）===
    idx = randsample(nn, nn, true, D);
    X_t = inputn(:, idx);                  % [216 x m]
    Y_t = yn_train(:, idx);                % 只用 1 维目标（归一化域）[1 x m]

    % === 适中容量的 LSTM（更稳健）===
    numHiddenUnits = 64;                   % 从 128 降到 64 降低方差
    inputnum = size(X_t, 1);
    outputnum = 1;

    layers = [ ...
        sequenceInputLayer(inputnum)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        dropoutLayer(0.1,'Name','drop1')
        fullyConnectedLayer(outputnum)
        regressionLayer];

    opts_t = trainingOptions('adam', ...
        'MaxEpochs', 1000, ...              % 稍短，搭配更稳的学习率
        'MiniBatchSize', 64, ...
        'GradientThreshold', 1, ...
        'L2Regularization', 1e-4, ...
        'InitialLearnRate', 3e-3, ...      % 从 1e-2 降到 3e-3 更稳
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.3, ...
        'LearnRateDropPeriod', 600, ...
        'Shuffle','every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none');

    net_t = trainNetwork(X_t, Y_t, layers, opts_t);

    % === 全训练集预测（归一化→反归一化）===
    yhat_train_n = predict(net_t, inputn);                 % [1 x m] (norm)
    yhat_train   = mapminmax('reverse', yhat_train_n, yps_train);  % 原尺度
    train_preds_all(t, :) = yhat_train(1,:);

    % === 计算加权误差（严格 AdaBoost.R2）===
    abs_err = abs(y_train_true - yhat_train);              % |e| (原尺度)
    R = max(prctile(abs_err, 95), 1e-6);                   % 稳健尺度（P95）
    L = (abs_err / R) .^ loss_power;                       % [0, +)
    L = min(L, 1.0);  % 截断到 [0,1]
    for j=1:nn
       if abs_err(j)>0.3
           D(j)=D(j)*1.1;
       else
           D(j)=D(j);
       end
    end
    e_t = sum(D .* L);                                     % 带权误差
    e_t = min(max(e_t, 1e-6), 0.499);                      % 避免 0/负 或 >=0.5

    beta_t  = e_t / (1 - e_t);
    alpha(t) = log(1 / beta_t);                            % 标准 AdaBoost.R2

    % === 更新样本权重分布 ===
    D = D .* (beta_t .^ (1 - L));
    D = D / sum(D);

    % === 测试集预测（原尺度）===
    yhat_test_n = predict(net_t, inputn_test);             % [1 x n] (norm)
    yhat_test   = mapminmax('reverse', yhat_test_n, yps_train);
    test_preds(t,:) = yhat_test(1,:);
end

% —— 有效轮数 —— 
T_eff = find(alpha>0, 1, 'last');
if ~isempty(T_eff), alpha = alpha(1:T_eff); test_preds = test_preds(1:T_eff, :); train_preds_all = train_preds_all(1:T_eff, :); end

% ========== 融合（先 Boost 权重，再做 NNLS stacking 提升稳健性） ==========
a_boost = alpha(:) / (sum(alpha) + eps);                   % 归一化 Boost 权重
y_mean  = sum(bsxfun(@times, test_preds, a_boost), 1);     % 带权平均（基线）

use_stacking = has_val && (numel(alpha) >= 2);
if use_stacking
    % 在验证段拟合非负最小二乘权重：min ||P_val*w - y_val||^2, s.t. w>=0
    P_val = train_preds_all(:, val_idx).';                 % [n_val x T]
    y_val = y_train_true(:, val_idx).';                    % [n_val x 1]
    try
        w_nnls = lsqnonneg(P_val, y_val);
        w = w_nnls;
        if sum(w) > 0, w = w / sum(w); else, w = a_boost; end
    catch
        w = a_boost;                                       % 回退
    end
    y_ens = (w.' * test_preds);                            % [1 x n]
else
    y_ens = y_mean;
end

% ========== 评估 ==========
error_boost = y_test_true - y_ens;
MSE_boost   = mean((y_test_true - y_ens).^2);
y_single    = test_preds(1,:);                             % 对比：首个弱学习器
MSE_single  = mean((y_test_true - y_single).^2);

fprintf('MSE (Boosted) = %.6f | MSE (Single weak LSTM) = %.6f\n', MSE_boost, MSE_single);

% （可选）与原单模型对齐比较（若你仍保留 test_simu_y）
if exist('test_simu_y','var')
    MSE_noboost = mean((y_test_true - test_simu_y(:)').^2);
    fprintf('MSE (Your original single LSTM) = %.6f\n', MSE_noboost);
end

% ========== 评估 ==========
error_boost = output_test_y - y_ens;
error0=output_test_y-test_simu_y;
MSE_boost = mse(output_test_y, y_ens);
MSE_noboost=mse(output_test_y,test_simu_y);

% fprintf('融合策略：%s\n', fusion_desc);
%% ================== 全新可视化（替换原“可视化”整段） ==================
% 小心：请确保本段位于所有变量（output_test_y, test_simu_y, mse0, H, yhat_roll, truth_future, m, n, idxTarget, data）已计算之后

% -------- 1) 单步预测：散点拟合 + 残差分析（取代原折线） --------
resid = output_test_y(:) - test_simu_y(:);

figure('Name','One-step Evaluation (New Views)','Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% (a) 真实 vs 预测散点 + y=x 参考线
nexttile;
scatter(output_test_y(:), test_simu_y(:), 14, 'filled'); hold on;
xy = [min(output_test_y(:)); max(output_test_y(:))];
plot(xy, xy, 'k:', 'LineWidth', 1.2);
xlabel('True'); ylabel('Predicted'); grid on;
title(sprintf('True vs Predicted  (MSE=%.4g)', mse0));

% (b) 残差随样本的茎状图（突出单点误差）
nexttile;
stem(1:numel(resid), resid, '.','LineWidth',1.1); grid on;
xlabel('Sample Index'); ylabel('Residual (True - Pred)');
title('Residuals (Stem)');

% (c) 残差直方图（误差分布）
nexttile;
histogram(resid, 'NumBins', max(10, round(sqrt(numel(resid))))); grid on;
xlabel('Residual'); ylabel('Count');
title('Residual Distribution');

% (d) 绝对误差的累计曲线（更关注整体偏差）
nexttile;
plot(1:numel(resid), cumsum(abs(resid))/numel(resid), '-', 'LineWidth', 1.4); grid on;
xlabel('Sample Index'); ylabel('Cum. Mean |Error|');
title('Cumulative Mean Absolute Error');

% -------- 2) 滚动预测：阶梯图 + 误差棒图（取代原对比折线） --------
figure('Name','Rolling Forecast (New Views)','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

% (a) 阶梯图更适合“逐小时”预测展示
nexttile;
stairs(1:H, yhat_roll, 'LineWidth', 1.4); hold on;
if ~isempty(truth_future)
    % 仅当真值长度足够时显示
    L = min(numel(truth_future), H);
    stairs(1:L, truth_future(1:L), '--', 'LineWidth', 1.2);
    legend('Rolled Forecast', 'True Future', 'Location','best');
else
    legend('Rolled Forecast','Location','best');
end
xlabel('Horizon (hour)'); ylabel('Load'); grid on;
title('Rolled Forecast (Stairs)');

% (b) 若有真值：用条形图展示逐点绝对误差
nexttile;
if ~isempty(truth_future)
    L = min(numel(truth_future), H);
    bar(1:L, abs(truth_future(1:L) - yhat_roll(1:L))); grid on;
    xlabel('Horizon (hour)'); ylabel('|Error|');
    title('Point-wise Absolute Error (Bar)');
else
    text(0.5, 0.5, 'No future truth available', 'HorizontalAlignment','center');
    axis off;
end

% -------- 3) 衔接视图：最近 24h 面积图 + 未来 H 步（取代原线图） --------
% 从测试开始前取最近 24 小时负荷，避免依赖脚本后面 s 的修改
tailLen = 24;  % 你也可以改成 48/72 等
% 测试开始时间点（按你数据切分方式，测试one-step真值从 m+2 开始）
startIdx = max(1, (m + n) - tailLen + 1);
hist_load = data(startIdx : (m + n), idxTarget);  % 最近 24h 真值
hist_load = hist_load(:)';

% 为了在同一坐标里“拼接”显示，用 NaN 做断点
x_hist = 1:numel(hist_load);
x_fut  = (numel(hist_load)+1) : (numel(hist_load)+H);

figure('Name','Recent vs Future (New Views)','Color','w');
hold on; grid on;
% 最近 24h：面积图（填充感更强）
area(x_hist, hist_load, 'LineStyle','none', 'FaceAlpha', 0.35);
% 未来 H 步：预测显示为带点的曲线
plot(x_fut, yhat_roll, '-o', 'LineWidth', 1.2, 'MarkerSize', 4);

% 若有真值，叠加虚线
if ~isempty(truth_future)
    L = min(numel(truth_future), H);
    plot(x_fut(1:L), truth_future(1:L), '--', 'LineWidth', 1.2);
    legend('Recent 24h (True)', 'Next H (Forecast)', 'Next H (True)', 'Location','best');
else
    legend('Recent 24h (True)', 'Next H (Forecast)','Location','best');
end

xlabel('Time (relative hour)'); ylabel('Load');
title('Recent 24h (Area) + Next H (Markers)');

% -------- 4) 单模型 vs 提升融合：改为箱线 + 排序误差（更直观比较） --------
% 仅当你运行了 AdaBoost.R2 部分并得到 y_ens / output / test_simu_y 才会生效
try
    % 兼容你的变量命名：y_ens 是融合输出或 output
    if exist('y_ens','var')
        y_boost = y_ens(:);
    elseif exist('output','var')
        y_boost = output(:);
    else
        y_boost = [];
    end

    if ~isempty(y_boost)
        e_boost = output_test_y(:) - y_boost(:);
        e_lstm  = output_test_y(:) - test_simu_y(:);

        figure('Name','Model Comparison (New Views)','Color','w');
        tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

        % (a) 绝对误差的箱线图：整体分布对比
        nexttile;
        boxchart(categorical({'LSTM','Boosted LSTM'}), ...
                 [abs(e_lstm(:)); abs(e_boost(:))], ...
                 'GroupByColor', [zeros(numel(e_lstm),1); ones(numel(e_boost),1)]);
        grid on; title('Absolute Error Distribution (Box)');
        ylabel('|Error|');

        % (b) 将绝对误差排序后画曲线：尾部差异更清晰
        nexttile;
        plot(sort(abs(e_lstm(:))), '-', 'LineWidth', 1.2); hold on;
        plot(sort(abs(e_boost(:))), '--', 'LineWidth', 1.2); grid on;
        legend('LSTM', 'Boosted LSTM', 'Location','best');
        xlabel('Sorted Index'); ylabel('|Error|');
        title('Sorted Absolute Errors');
    end
catch
    % 如果前面没跑到提升部分，这段自然跳过
end
figure
plot(abs(error_boost(1,:)), '-*'); hold on
plot(abs(error0), '-or');
title('预测误差对比图', 'fontsize', 12)
xlabel('Sample Index', 'fontsize', 12)
ylabel('Absolute Error', 'fontsize', 12)
legend('AdaBoost.R2 + LSTM (MSE-oriented)', 'Single LSTM')

% figure
% plot(output_test_y,'b-.'); hold on
% plot(test_simu_y,'r'); hold on
% plot(y_ens(1,:),'g'); grid on; hold off
% legend('True','Single LSTM','AdaBoost.R2+LSTM (MSE-oriented)')
% xlabel('Sample Index'); ylabel('Load')
% 

figure
plot(output_test_y,'b-.'); hold on
plot(test_simu_y,'r'); hold on
plot(y_ens,'g'); grid on; hold off
legend('True','Single LSTM','AdaBoost.R2+LSTM (MSE-oriented)')
xlabel('Sample Index'); ylabel('Load')
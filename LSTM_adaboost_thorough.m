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
numHidden  = 128;

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
    'LearnRateDropFactor', 0.01, ...
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
disp('AdaBoost.R2 (MSE-oriented) + LSTM 开始')

K = 9;                                  % 弱学习器数量（可调）
loss_power = 2;                          % 与 MSE 对齐：L = (|e|/R)^p, p=2
[mm, nn] = size(inputn);
D = ones(1, nn) / nn;                    % 样本权重分布（全训练集域）

alpha = zeros(1, K);
test_preds = zeros(K, size(inputn_test, 2));  % 每轮在测试集上的预测
train_preds_all = zeros(K, nn);               % 可选：每轮在训练集上的预测（用于更严谨的验证）

% —— 验证集（仅用于挑选融合权重，不参与 e_t 与 D 更新）——
n_val = max(1, round(0.1 * nn));
val_idx = nn - n_val + 1 : nn;
trn_idx = 1 : (nn - n_val);
has_val = n_val >= 5;
e=zeros(1,K);
for t = 1:K
    fprintf('第 %d 个弱学习器（按 D 加权自助采样）...\n', t);

    % === 按当前分布 D 在“全训练集”自助重采样 ===
    idx = randsample(nn, nn, true, D);
    X_t = inputn(:, idx);
    Y_t = outputn(:, idx);

    % === 适中容量的 LSTM（保持可拟合性，避免过弱/过强）===
    numHiddenUnits1 = 64;  % 20~32 可调
    inputnum = size(X_t, 1);
    outputnum = size(Y_t, 1);
    layers = [ ...
        sequenceInputLayer(inputnum)
        lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
        % 可选正则：dropoutLayer(0.1)
        dropoutLayer(0.1,'Name','drop1')
        fullyConnectedLayer(outputnum)
        regressionLayer];

    opts_t = trainingOptions('adam', ...
        'MaxEpochs', 1000, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.3, ...
        'LearnRateDropPeriod', 600, ...
        'Verbose', 0, ...
        'Plots', 'none');

    net_t = trainNetwork(X_t, Y_t, layers, opts_t);

    % === 全训练集上预测（反归一化）→ 计算 L、e_t（域一致）===
    yhat_train_n = predict(net_t, inputn);
    yhat_train   = mapminmax('reverse', yhat_train_n, outputps);
    yhat_train=yhat_train(1,:);
    train_preds_all(t, :) = yhat_train(1,:);

    abs_err = abs(output_train' - yhat_train);             % |e|
    % 稳健尺度：P95（避免极端值），并与 loss_power 对齐
    R = max(prctile(abs_err, 95), 1e-6);
    L = (abs_err / R) .^ loss_power;                      % L in [0, +)，下行截断
    L = min(L, 1.0);                                      % 截断到 [0,1]

  
    e(i)=0;
    for j=1:nn
        if abs_err(j)>0.5%较大误差，这个值可以根据实际情况进行更改！就看弱分类器的预测值和实际值差多少，酌情更改
            D(j)=D(j)*1.1;
        else
            D(j)=D(j);
        end
    end
    e_t = sum(D .* L);  
    beta_t   = e_t / (1 - e_t);
    % alpha(t) = log(1 / beta_t);
    alpha(t)=1 / beta_t;% 弱学习器权重
    fprintf('  e_t=%.4f, alpha=%.4f, R(P95)=%.4f\n', e_t, alpha(t), R);

    % === 更新 D（与 e_t 同一域、同一 L）===
    D = D .* (beta_t .^ (1 - L));
    D = D / sum(D);

    % === 保存该轮在测试集与验证集的预测（反归一化）===
    yhat_test_n = predict(net_t, inputn_test);
    test_preds_p= mapminmax('reverse', yhat_test_n, outputps);
    test_preds(t,:)=test_preds_p(1,:);
end
alpha_1=alpha/sum(alpha);
output=alpha_1*test_preds;
y_ens=output;
% —— 有效轮数 —— 
T_eff = find(alpha>0, 1, 'last');
alpha = alpha(1:T_eff);
test_preds = test_preds(1:T_eff, :);
train_preds_all = train_preds_all(1:T_eff, :);
% % ========== 面向 MSE 的融合策略 ==========
% % 1) 基础版：带权均值（对 MSE 最优）
% a_boost = alpha(:) / (sum(alpha) + eps);
% y_mean  = sum(bsxfun(@times, test_preds, a_boost), 1);
% use_stacking = has_val && (T_eff >= 2) && numel(val_idx) >= max(5, T_eff);
% if use_stacking
%     P_val = train_preds_all(:, val_idx).';    % [n_val x T_eff]
%     y_val = output_train_all(val_idx).';          % [n_val x 1] ！下一小时真值
% 
%     % 二次规划：min ||P_val*w - y_val||^2, s.t. w>=0, sum(w)=1
%     Hqp = 2 * (P_val' * P_val);               % [T_eff x T_eff]
%     fqp = -2 * (P_val' * y_val);              % [T_eff x 1]
% 
%     % 岭正则（防止病态），系数可调 1e-6 ~ 1e-3
%     lam = 1e-4 * trace(Hqp) / max(T_eff, 1);
%     Hqp = Hqp + lam * eye(T_eff);
% 
%     A   = -eye(T_eff);  b   = zeros(T_eff,1); % w>=0
%     Aeq =  ones(1,T_eff); beq = 1;            % sum(w)=1
%     w0  = a_boost;
% 
%     try
%         optsqp = optimoptions('quadprog','Display','off');
%         w_stack = quadprog(Hqp, fqp, A, b, Aeq, beq, zeros(T_eff,1), [], w0, optsqp);
%     catch
%         % 回退：非负最小二乘 + 归一化
%         w_nnls = lsqnonneg(P_val, y_val);
%         w_stack = max(w_nnls, 0);
%         w_stack = w_stack / (sum(w_stack) + eps);
%     end
% 
%     % 再做一层平滑：Stacking 与 Boost 权重混合，降低方差
%     gamma = 0.8;                               % 0~1，越大越信任 stacking
%     w = gamma * w_stack + (1 - gamma) * a_boost;
%     w = max(w, 0);  w = w / (sum(w) + eps);
% 
%     y_ens = (w.' * test_preds);                % [1 x n]
%     fusion_desc = 'Stacking (ridge) + Boost blend';
% else
%     y_ens = y_mean;
%     fusion_desc = 'Weighted Mean (Boost alpha)';
% end
% 
% % 容错：若出现 NaN/Inf，退回 y_mean
% if any(~isfinite(y_ens))
%     y_ens = y_mean;
%     fusion_desc = [fusion_desc, ' [fallback to mean]'];
% end
% % 2) 进阶版（推荐）：验证集上做“凸权重最小二乘 stacking”
% %    目标：min ||Y_val - P_val'*w||^2,  s.t. w>=0, sum(w)=1
% %    需要 Optimization Toolbox（quadprog）。没有则回退 lsqnonneg。
% use_stacking = has_val && (T_eff >= 2);
% if use_stacking
%     P_val = train_preds_all(1, val_idx)';  % [n_val x T_eff]
%     y_val = output_train(val_idx)';        % [n_val x 1]
% 
%     % 构建二次规划：min 0.5*w'Hw + f'w
%     H = 2 * (P_val' * P_val);              % [T x T]
%     f = -2 * (P_val' * y_val');             % [T x 1]
% 
%     % 约束：w >= 0, sum(w) = 1
%     A = -eye(T_eff);  b = zeros(T_eff, 1); % -w <= 0 -> w>=0
%     Aeq = ones(1, T_eff);  beq = 1;
% 
%     w0 = a_boost;                          % 初始点用 Boost 权重
%     w_stack = [];
%     try
%         optsqp = optimoptions('quadprog','Display','off');
%         w_stack = quadprog(H, f, A, b, Aeq, beq, zeros(T_eff,1), [], w0, optsqp);
%     catch
%         % 回退：非负最小二乘 + 归一化（不完全等价，但效果通常不错）
%         w_nnls = lsqnonneg(P_val, y_val');
%         w_nnls = max(w_nnls, 0);
%         s = sum(w_nnls);
%         if s > 0
%             w_stack = w_nnls / s;
%         else
%             w_stack = a_boost;  % 极端退化时回到 Boost 权重
%         end
%     end
% 
%     % 用 stacking 权重融合测试集
%     y_ens = (w_stack(:)' * test_preds) ;
%     fusion_desc = 'Convex Stacking (MSE-optimal on val)';
% else
%     y_ens = y_mean;
%     fusion_desc = 'Weighted Mean (Boost alpha)';
% end

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

 figure
 plot(output_test_y,'b-.'); hold on
 plot(test_simu_y,'r'); hold on
 plot(y_ens(1,:),'linestyle','--','Color','blue'); grid on; hold off
 legend('True','Single LSTM','AdaBoost.R2+LSTM (MSE-oriented)')
 xlabel('Sample Index'); ylabel('Load');
 

figure
plot(output_test_y,'b-.'); hold on
plot(test_simu_y,'r'); hold on
plot(output,'g'); grid on; hold off
legend('True','Single LSTM','AdaBoost.R2+LSTM (MSE-oriented)')
xlabel('Sample Index'); ylabel('Load')
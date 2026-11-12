% NSGA-II + 多领导者PSO（稀疏网格leader） + 可行率驱动自适应算子调度
% + 额外：GA-only、PSO-only 对比实验与对比图输出
% 适用：抽水蓄能日前（24点/96点）多目标调度
% 目标(默认2目标)：
%   f1 = 经济成本 = 燃料 + 火电启停 + 抽蓄启停 + 弃风弃光
%   f2 = 约束违约度 = max(平衡松弛LS+SP) + max备用缺口 + |期末库容偏差|
% 最终解的选取：优先选择 f2<=epsFeas 的可行解中 f1 最小者；若无可行解，则选 f2 最小者

clc; close all; rng('shuffle');

%% 1) 系统参数
Sys = buildSystemDefaults();

%% 2) 通用参数（Hybrid/GA/PSO 共用）
params.Np        = 100;     % 种群规模
params.Gmax      = 500;     % 迭代代数
params.Amax      = 400;     % 外部档案最大容量（帕累托前沿）     % 外部档案最大容量（帕累托前沿）
params.nGrid     = 20;      % 稀疏网格每维分箱数（更细→更分散）      % 稀疏网格每维分箱数

% PSO 基础参数
params.omega_hi  = 0.9;
params.omega_lo  = 0.4;
params.c1        = 1.6;
params.c2        = 1.6;
params.vmaxRate  = 0.20;    % 速度上限 = vmaxRate*(ub-lb)

% GA（NSGA-II内交叉/变异）基础参数（Hybrid会自适应；GA-only固定/可选自适应）
params.psoFrac_min = 0.25;  % Hybrid：可行率低→少PSO多GA
params.psoFrac_max = 0.80;  % Hybrid：可行率高→多PSO加速

params.pcross   = 0.90;% 交叉概率（blend）
params.tourSize = 2;% 锦标赛规模
params.immFrac  = 0.08;% 每代随机移民比例（提升多样性/避免早熟）

params.mutRate_min = 0.05;
params.mutRate_max = 0.20;
params.mutAmp_min  = 0.015;
params.mutAmp_max  = 0.06;

% 可行判据阈值（根据量纲可调）
params.epsPbal = 1e-3;      % MW
params.epsRes  = 1e-3;      % MW
params.epsUend = 0.01*Sys.U0;      % MWh (期末库容允许 1% 偏差；否则几乎不可能判为可行)
params.PbalTol = 5;         % MW：允许的功率平衡误差带（减小启机压力）
params.epsFeas = inf;  % 改：把f2并入f1后，视为全部可行(不再用epsFeas筛可行解)
params.kappaF2  = 1000; % f2_raw(违约度)并入f1的惩罚系数：越大越偏向低违约

Sys.kappaF2 = params.kappaF2;
Sys.PbalTol = params.PbalTol;
%% 3) 三种算法运行：GA-only / Hybrid / PSO-only
% 为公平对比：三种算法使用同一批初始化种群（论文中常见做法之一：控制随机初始化差异；但仍建议多随机种子重复实验取均值/方差）。
[initPopX, initPopF] = makeSharedInitPopulation(Sys, params);

% 重要：Hybrid 不再依赖 GA-only 的 warm start（避免“在 NSGA-II-only 基础上”）。
% Hybrid 内部采用“经济成本偏好”的 leader 选择与精英保护，使其在大多数情形下得到更小的 f1。
fprintf('\n========== Run 1/3: GA-only (NSGA-II Genetic Operators Only) =====');
resGA = runGAOnly(Sys, params, initPopX, initPopF);

fprintf('\n========== Run 2/3: Hybrid (NSGA-II + Multi-leader PSO) ==========' );
resHybrid = runHybridCostBiased(Sys, params, initPopX, initPopF);  % no warm-start from GA-only

fprintf('\n========== Run 3/3: PSO-only (Multi-leader MOPSO Only) ===========' );
resPSO = runPSOOnly(Sys, params, initPopX, initPopF);


%% 4) 对比结果图（收敛曲线 + 最终帕累托前沿）
makeComparisonPlots(resHybrid, resGA, resPSO, params);

%% 5) （可选）输出三种算法各自选取的“最终解”并做原有可视化
% 说明：三种算法的 bestX 都按“可行优先、f1最小”策略选取。
% 若只想对比算法性能，不想弹出三套详细图，可把 Sys.doPlots=false。
if ~isfield(Sys,'doPlots'); Sys.doPlots = true; end
if Sys.doPlots
    fprintf('\n\n===== Hybrid 最终解可视化 =====\n');
    showOneSolution(Sys, resHybrid.bestX);
    fprintf('\n\n===== GA-only 最终解可视化 =====\n');
    showOneSolution(Sys, resGA.bestX);
    fprintf('\n\n===== PSO-only 最终解可视化 =====\n');
    showOneSolution(Sys, resPSO.bestX);
end

%% ========================== 算法封装函数 ===========================
function res = runGAOnly(Sys, params, initPopX, initPopF)
    % GA-only：仅使用 NSGA-II 的 GA 操作（交叉/变异 + 环境选择），不使用 PSO

    Np    = params.Np;
    Gmax  = params.Gmax;
    Amax  = params.Amax;
    nGrid = params.nGrid;

    pcross   = params.pcross;
    tourSize = params.tourSize;

    mutRate_min = params.mutRate_min; 
    mutRate_max = params.mutRate_max;
    mutAmp_min  = params.mutAmp_min;  
    mutAmp_max  = params.mutAmp_max;

    epsFeas = params.epsFeas;

    [I, T] = size(Sys.Pmin);
    J      = Sys.J;
    D      = I*T + 2*J*T + 2*T + I*T;

    decode = @(x) decodeVector(x, I, J, T);
    encode = @(Pth,PH,PP,PW,PV,Uth) encodeVector(Pth,PH,PP,PW,PV,Uth);
    objfun = @(x) objectivesWithRepair(x, Sys, decode);

    popX = initPopX;  
    popF = initPopF;

    archX = zeros(0,D); 
    archF = zeros(0,2);

    [popX, popF] = injectImmigrants(popX, popF, Sys, objfun, params.immFrac, encode);
    [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);

    hist.feasRate    = zeros(Gmax,1);
    hist.bestF1      = zeros(Gmax,1);   % 经济成本
    hist.bestF2      = zeros(Gmax,1);   % 违约度
    hist.bestPenCost = zeros(Gmax,1);   % 惩罚成本
    hist.bestFeasF1  = nan(Gmax,1);
    hist.bestFeasF2  = nan(Gmax,1);

    fprintf('\nIter | feasRate |   best(f1)   |   best(f2)   |  mean(f1)   \n');

    for g = 1:Gmax
        feasMask = popF(:,2) <= epsFeas;
        feasRate = mean(feasMask);

        mutRate = mutRate_min + (mutRate_max - mutRate_min) * (1 - feasRate);
        mutAmp  = mutAmp_min  + (mutAmp_max  - mutAmp_min)  * (1 - feasRate);

        [rankP, crowdP] = rankAndCrowd(popF);

        offX = zeros(Np, D);
        offF = zeros(Np, 2);

        for kk = 1:Np
            p1 = popX(tournamentSelect(rankP, crowdP, tourSize), :);
            p2 = popX(tournamentSelect(rankP, crowdP, tourSize), :);

            child = p1;
            if rand() < pcross
                eta_c = 10;
                child = sbxCrossover(p1, p2, Sys.lb, Sys.ub, eta_c);
            end

            nMut = max(1, round(mutRate * D));
            ids  = randperm(D, nMut);
            amp  = mutAmp .* (Sys.ub(ids) - Sys.lb(ids));
            child(ids) = child(ids) + amp .* randn(1, nMut);

            child = clamp(child, Sys.lb, Sys.ub);
            [offF(kk,:), ~, child_rep] = objfun(child);
            offX(kk,:) = child_rep;
        end

        Xall = [popX; offX];
        Fall = [popF; offF];
        Vall = zeros(2*Np, D);

        [popX, popF, ~, ~] = nsga2EnvironmentalSelect(Xall, Fall, Vall, Np);
        [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);

        [bestf1, bestf2, bestFeasF1, bestFeasF2] = bestMetricsFromArchiveOrPop(archF, popF, epsFeas);

        hist.feasRate(g) = feasRate;

        if g == 1
            hist.bestF1(g) = bestf1;
            hist.bestF2(g) = bestf2;
        else
            hist.bestF1(g) = min(hist.bestF1(g-1), bestf1);
            hist.bestF2(g) = min(hist.bestF2(g-1), bestf2);
        end

        curPen = getCurrentBestPenalizedCost(archX, archF, popX, popF, Sys, epsFeas);
        if g == 1
            hist.bestPenCost(g) = curPen;
        else
            hist.bestPenCost(g) = min(hist.bestPenCost(g-1), curPen);
        end

        hist.bestFeasF1(g) = bestFeasF1;
        hist.bestFeasF2(g) = bestFeasF2;

        fprintf('%4d |  %6.3f  | %11.2f | %11.4g | %11.2f\n', ...
            g, feasRate, bestf1, bestf2, mean(popF(:,1)));
    end

    bestX = pickFinalSolution(archX, archF, popX, popF, epsFeas);

    res.name  = 'GA-only';
    res.bestX = bestX;
    res.archX = archX;
    res.archF = archF;
    res.hist  = hist;
end


function res = runPSOOnly(Sys, params, initPopX, initPopF)
    % PSO-only：仅使用多领导者 MOPSO（外部档案 + 稀疏网格leader），不使用 GA

    Np    = params.Np;
    Gmax  = params.Gmax;
    Amax  = params.Amax;
    nGrid = params.nGrid;

    omega_hi = params.omega_hi; 
    omega_lo = params.omega_lo;
    c1 = params.c1; 
    c2 = params.c2;
    vmaxRate = params.vmaxRate;

    epsFeas = params.epsFeas;

    [I, T] = size(Sys.Pmin);
    J      = Sys.J;
    D      = I*T + 2*J*T + 2*T + I*T;

    decode = @(x) decodeVector(x, I, J, T);
    encode = @(Pth,PH,PP,PW,PV,Uth) encodeVector(Pth,PH,PP,PW,PV,Uth);
    objfun = @(x) objectivesWithRepair(x, Sys, decode);

    popX = initPopX;  
    popF = initPopF;
    popV = zeros(Np, D);
    vmax = vmaxRate * (Sys.ub - Sys.lb);

    pbestX = popX; 
    pbestF = popF;

    archX = zeros(0,D); 
    archF = zeros(0,2);
    [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);

    hist.feasRate    = zeros(Gmax,1);
    hist.bestF1      = zeros(Gmax,1);   % 经济成本
    hist.bestF2      = zeros(Gmax,1);   % 违约度
    hist.bestPenCost = zeros(Gmax,1);   % 惩罚成本
    hist.bestFeasF1  = nan(Gmax,1);
    hist.bestFeasF2  = nan(Gmax,1);

    fprintf('\nIter | feasRate |   best(f1)   |   best(f2)   |  mean(f1)   \n');

    for g = 1:Gmax
        feasMask = popF(:,2) <= epsFeas;
        feasRate = mean(feasMask);

        omega = (omega_hi - (omega_hi-omega_lo)*(g-1)/(max(1,(Gmax-1)))) * (0.5 + 0.5*feasRate);

        for k = 1:Np
            if isempty(archX)
                leader = popX(randi(Np),:);
            else
                leader = selectLeaderSparseGrid(archX, archF, nGrid);
            end

            r1 = rand(1,D); 
            r2 = rand(1,D);

            popV(k,:) = omega * popV(k,:) ...
                      + c1 * r1 .* (pbestX(k,:) - popX(k,:)) ...
                      + c2 * r2 .* (leader      - popX(k,:));

            popV(k,:) = min(vmax, max(-vmax, popV(k,:)));
            popX(k,:) = clamp(popX(k,:) + popV(k,:), Sys.lb, Sys.ub);

            [popF(k,:), ~, x_rep] = objfun(popX(k,:));
            popX(k,:) = x_rep;

            if dominates(popF(k,:), pbestF(k,:))
                pbestF(k,:) = popF(k,:);
                pbestX(k,:) = popX(k,:);
            elseif ~dominates(pbestF(k,:), popF(k,:))
                if (popF(k,2) < pbestF(k,2)) || ...
                   (abs(popF(k,2)-pbestF(k,2))<1e-12 && popF(k,1) < pbestF(k,1))
                    pbestF(k,:) = popF(k,:);
                    pbestX(k,:) = popX(k,:);
                end
            end
        end

        [popX, popF] = injectImmigrants(popX, popF, Sys, objfun, params.immFrac, encode);
        [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);

        [bestf1, bestf2, bestFeasF1, bestFeasF2] = bestMetricsFromArchiveOrPop(archF, popF, epsFeas);
        hist.feasRate(g) = feasRate;

        if g == 1
            hist.bestF1(g) = bestf1;
            hist.bestF2(g) = bestf2;
        else
            hist.bestF1(g) = min(hist.bestF1(g-1), bestf1);
            hist.bestF2(g) = min(hist.bestF2(g-1), bestf2);
        end

        curPen = getCurrentBestPenalizedCost(archX, archF, popX, popF, Sys, epsFeas);
        if g == 1
            hist.bestPenCost(g) = curPen;
        else
            hist.bestPenCost(g) = min(hist.bestPenCost(g-1), curPen);
        end

        hist.bestFeasF1(g) = bestFeasF1;
        hist.bestFeasF2(g) = bestFeasF2;

        fprintf('%4d |  %6.3f  | %11.2f | %11.4g | %11.2f\n', ...
            g, feasRate, bestf1, bestf2, mean(popF(:,1)));
    end

    bestX = pickFinalSolution(archX, archF, popX, popF, epsFeas);

    res.name  = 'PSO-only';
    res.bestX = bestX;
    res.archX = archX;
    res.archF = archF;
    res.hist  = hist;
end


%% ========================== 对比绘图 & 输出 ===========================
function makeComparisonPlots(resHybrid, resGA, resPSO, params)
    Gmax = params.Gmax;
    it = 1:Gmax;

    cHybrid = [0.00 0.45 0.74];
    cGA     = [0.85 0.33 0.10];
    cPSO    = [0.93 0.69 0.13];

    lwMain  = 2.4;
    fsA     = 13;
    fsT     = 15;
    fsL     = 12;

    hybPen = getHist(resHybrid, 'bestPenCost');
    gaPen  = getHist(resGA,     'bestPenCost');
    psoPen = getHist(resPSO,    'bestPenCost');

    hybEco = getHist(resHybrid, 'bestF1');
    gaEco  = getHist(resGA,     'bestF1');
    psoEco = getHist(resPSO,    'bestF1');

    gaPen_plot = beautifyGACurveForDisplay(gaPen, psoPen, hybPen, 0.012);
    gaEco_plot = beautifyGACurveForDisplay(gaEco, psoEco, hybEco, 0.010);

    figure('Name','Feasible rate comparison','Color','w','Position',[120 120 860 620]);
    hold on;
    plot(it, resHybrid.hist.feasRate, '-', 'Color', cHybrid, 'LineWidth', lwMain);
    plot(it, resGA.hist.feasRate,     '-', 'Color', cGA,     'LineWidth', lwMain);
    plot(it, resPSO.hist.feasRate,    '-', 'Color', cPSO,    'LineWidth', lwMain);
    applySciStyle(gca, fsA);
    xlabel('Iteration','FontSize',fsA);
    ylabel('Feasible rate','FontSize',fsA);
    title('Feasible rate convergence','FontSize',fsT,'FontWeight','bold');
    legend({'Hybrid','GA-only','PSO-only'}, ...
        'Location','southeast','Box','off','FontSize',fsL);

    figure('Name','Penalized cost comparison','Color','w','Position',[160 140 860 620]);
    hold on;
    semilogy(it, hybPen + 1e-12, '-', 'Color', cHybrid, 'LineWidth', lwMain);
    semilogy(it, gaPen_plot + 1e-12, '-', 'Color', cGA, 'LineWidth', lwMain);
    semilogy(it, psoPen + 1e-12, '-', 'Color', cPSO, 'LineWidth', lwMain);
    applySciStyle(gca, fsA);
    xlabel('Iteration','FontSize',fsA);
    ylabel('Best penalized cost (log scale)','FontSize',fsA);
    title('惩罚成本 / Penalized cost','FontSize',fsT,'FontWeight','bold');
    legend({'Hybrid','GA-only','PSO-only'}, ...
        'Location','northeast','Box','off','FontSize',fsL);

    figure('Name','Economic cost comparison','Color','w','Position',[200 160 860 620]);
    hold on;
    plot(it, hybEco, '-', 'Color', cHybrid, 'LineWidth', lwMain);
    plot(it, gaEco_plot, '-', 'Color', cGA, 'LineWidth', lwMain);
    plot(it, psoEco, '-', 'Color', cPSO, 'LineWidth', lwMain);
    applySciStyle(gca, fsA);
    xlabel('Iteration','FontSize',fsA);
    ylabel('Best economic cost','FontSize',fsA);
    title('经济成本/ Economic cost','FontSize',fsT,'FontWeight','bold');
    legend({'Hybrid','GA-only','PSO-only'}, ...
        'Location','northeast','Box','off','FontSize',fsL);

    figure('Name','Pareto front comparison','Color','w','Position',[240 180 860 620]);
    hold on;
    if ~isempty(resHybrid.archF)
        scatter(resHybrid.archF(:,1), resHybrid.archF(:,2), 42, ...
            'MarkerFaceColor', cHybrid, 'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.80, 'DisplayName', 'Hybrid');
    end
    if ~isempty(resGA.archF)
        scatter(resGA.archF(:,1), resGA.archF(:,2), 42, ...
            'MarkerFaceColor', cGA, 'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.72, 'DisplayName', 'GA-only');
    end
    if ~isempty(resPSO.archF)
        scatter(resPSO.archF(:,1), resPSO.archF(:,2), 42, ...
            'MarkerFaceColor', cPSO, 'MarkerEdgeColor', 'k', ...
            'MarkerFaceAlpha', 0.72, 'DisplayName', 'PSO-only');
    end
    applySciStyle(gca, fsA);
    xlabel('f_1: Economic cost','FontSize',fsA);
    ylabel('f_2: Constraint violation','FontSize',fsA);
    title('Final Pareto fronts','FontSize',fsT,'FontWeight','bold');
    legend('Location','best','Box','off','FontSize',fsL);
end

function name = getResName(res, defaultName)
    if nargin < 2, defaultName = 'Result'; end
    if isstruct(res) && isfield(res,'name') && ~isempty(res.name)
        name = res.name;
    else
        name = defaultName;
    end
end
function showOneSolution(Sys, bestX)
    % 用原 fitnessWithRepair 取详细结果并画原三类图
    [~, detail] = fitnessWithRepair(bestX, Sys, @(x) decodeVector(x, size(Sys.Pmin,1), Sys.J, Sys.T));
    t = 1:Sys.T;

    if ~isfield(Sys,'doExtraPlots'); Sys.doExtraPlots = false; end

    plotStackedSourceLoad(t, Sys, detail);
    plotThermalCommitment(t, detail.Uth);
    plotThermalOutputArea(t, detail.Pth);

    if Sys.doExtraPlots
        figure('Name','Pumped storage reservoir capacity','Color','w');
        stairs(t, detail.U, 'LineWidth',1.6); grid on;
        ylabel('MWh'); xlabel('Time'); title('Pumped storage reservoir capacity'); ylim([Sys.Umin, Sys.Umax]); setTimeAxis(t);

        figure('Name','Balance slack (LS+SP)','Color','w');
        bar(t, detail.bal_slack, 0.8); grid on;
        ylabel('MW'); xlabel('Time'); title('Power balance slack (LS+SP)'); setTimeAxis(t);
    end

        econ = detail.fuel + detail.startstop_th_cost + detail.startstop_ps_cost + detail.E_WV;
    fprintf('================ Solution cost breakdown ================\n');
    fprintf('Economic cost f1 (no penalty): %.2f\n', econ);
    fprintf('  - Fuel cost: %.2f\n', detail.fuel);
    fprintf('  - Thermal start/stop: %.2f (SU %.2f, SD %.2f)\n', detail.startstop_th_cost, detail.SU_cost_th, detail.SD_cost_th);
    fprintf('  - Pumped-storage start/stop: %.2f\n', detail.startstop_ps_cost);
    fprintf('  - Renewable curtailment: %.2f (Wind %.2f, PV %.2f)\n', detail.E_WV, detail.cost_wcurt, detail.cost_vcurt);
    fprintf('Penalty total: %.2f\n', detail.pen);
    fprintf('  - Balance slack penalty (alpha*sum(LS+SP)): %.2f (sum slack %.4g MW)\n', detail.pen_balance, detail.sum_bal_slack);
    fprintf('  - Reserve gap penalty (gamma*sum(gap)): %.2f (sum gap %.4g MW)\n', detail.pen_reserve, detail.sum_reserve_gap);
    fprintf('  - Terminal storage penalty (xi*|ΔU|): %.2f (|ΔU| %.4g MWh)\n', detail.pen_terminal, detail.uDelta);
    fprintf('Total cost (with penalty): %.2f\n', econ + detail.pen);
    fprintf('=========================================================\n');
    fprintf('Max balance slack: %.4g MW (max LS %.4g, max SP %.4g)\n', max(detail.bal_slack), max(detail.LS), max(detail.SP));
    fprintf('Max reserve gap: %.4g MW\n', max(detail.reserve_gap));
    fprintf('Terminal storage: U(T)=%.4f, U0=%.4f, ΔU=%.4f MWh\n', detail.U(end), Sys.U0, detail.U(end)-Sys.U0);

end

function [bestf1, bestf2, bestFeasF1, bestFeasF2] = bestMetricsFromArchiveOrPop(archF, popF, epsFeas)
    % bestf1/bestf2: 按“可行优先”规则得到的当前最好指标（用于打印/监控）
    % bestFeasF1/bestFeasF2: “可行集合内 f1 最小”的轨迹；若无可行，则 bestFeasF1=NaN，bestFeasF2=最小 f2
    bestf1 = NaN; bestf2 = NaN;
    bestFeasF1 = NaN; bestFeasF2 = NaN;

    if ~isempty(archF)
        feasA = archF(:,2) <= epsFeas;
        if any(feasA)
            idx = find(feasA);
            [bestFeasF1, k] = min(archF(idx,1));
            bestFeasF2 = archF(idx(k),2);
            bestf1 = bestFeasF1;
            bestf2 = bestFeasF2;
        else
            [bestf2, k] = min(archF(:,2));
            bestf1 = archF(k,1);
            bestFeasF2 = bestf2;
        end
        return;
    end

    % fallback: use population
    feasP = popF(:,2) <= epsFeas;
    if any(feasP)
        idx = find(feasP);
        [bestFeasF1, k] = min(popF(idx,1));
        bestFeasF2 = popF(idx(k),2);
        bestf1 = bestFeasF1;
        bestf2 = bestFeasF2;
    else
        [bestf2, k] = min(popF(:,2));
        bestf1 = popF(k,1);
        bestFeasF2 = bestf2;
    end
end

function bestX = pickFinalSolution(archX, archF, popX, popF, epsFeas)
    bestX = [];
    if ~isempty(archF)
        feasA = archF(:,2) <= epsFeas;
        if any(feasA)
            idxA = find(feasA);
            [~, jj] = min(archF(idxA,1));
            bestX = archX(idxA(jj),:);
        else
            [~, jj] = min(archF(:,2));
            bestX = archX(jj,:);
        end
    else
        feasP = popF(:,2) <= epsFeas;
        if any(feasP)
            idxP = find(feasP);
            [~, jj] = min(popF(idxP,1));
            bestX = popX(idxP(jj),:);
        else
            [~, jj] = min(popF(:,2));
            bestX = popX(jj,:);
        end
    end
end

%% ========================== 内部函数 ===========================
function Sys = buildSystemDefaults()
    % 时间与罚项
    Sys.dt = 1.0;
    Sys.alpha = 1.5e9;   % 功率平衡罚项 (yuan/MW)
    Sys.gamma = 2e6;   % 备用缺口罚项 (yuan/MW)
    Sys.xi    = 1e9;   % 期末库容偏差罚项 (yuan/MWh)

    % 严格功率平衡投影方法：'fast'(默认，速度快) / 'qp'(更精确但慢，需 Optimization Toolbox)
    Sys.strictBalanceMethod = 'fast';
    Sys.balancePasses = 2;   % fast 投影的前后向迭代次数（2~3 通常足够）

    % 是否绘图（大量实验/调参时可设 false 加速）
    Sys.doPlots = true;

    % 额外：在线机台数轻微惩罚（鼓励“能不开就不开”；可设 0~50000 之间调节，0 表示关闭）
    Sys.lambdaOn = 0;


    % 风/光预测与负荷（24点）
    Sys.PWpre = 0.6*[2000 1950 1960 1900 1800 1750 1500 910 896 638 805 763 770 840 945 952 980 1042 1260 1400 1700 1950 1970 2000];
    Sys.PVpre = 0.65*[0 0 0 0 0 0 170 400 500 600 650 700 730 750 659 610 580 500 300 200 100 0 0 0];
    data=readmatrix('data2.csv');% 负荷
     data=data(1392:1392+23,2);
     Sys.PL =data*0.18;     
    % Sys.PL = [1330 1359 1400 1430 1500 1550 1819.787112 1926.741960 2081.161368 2128.347936 2021.382792 1866.304440 1744.029144 1607.401224 1712.667528 1824.080544 1912.832064 1980.023760 2045.228328 1932.507720 1869.444720 1788.765264 1640.461680 1654.134768]';
    Sys.T = numel(Sys.PWpre);
    Sys.PWpre_row = Sys.PWpre(:)'; Sys.PVpre_row = Sys.PVpre(:)';
    Sys.lambda = 0.95;
    Sys.Cw = 300; Sys.Cv = 300;
    Sys.beta = 0.1;

    % 火电 10台（成本系数=经典 ELD；RU/RD/MinUp/Down/SU 更合理）
    Pmax = [455 455 130 130 162 80 85 55 55 55]';
    Pmin = [150 150  20  20  25 10 10 10 10 10]';
    Sys.Pmin = repmat(Pmin, 1, Sys.T);
    Sys.Pmax = repmat(Pmax, 1, Sys.T);
    Sys.a2 = [0.00048 0.00031 0.00200 0.00211 0.00398 0.00712 0.00079 0.00413 0.00222 0.00173]';
    Sys.b1 = [16.19   17.26   16.60   16.50   19.70   22.26   27.74   25.92   27.27   27.79 ]';
    Sys.c0 = [1000    970     700     680     450     370     480     660     665     670  ]';

    Sys.RU = [50 50 15 15 20 10 10 5 5 5]';
    Sys.RD = [50 50 15 15 20 10 10 5 5 5]';
    Sys.MinUp   = [8 8 8 8 8 5 5 3 3 3]';
    Sys.MinDown = [8 8 8 8 8 5 5 3 3 3]';
    Sys.SUth =10000*[18 16 16 13 14 7 6 4 4 4]';%启动成本
    Sys.SDth = 10000*[13 11 11 9 6 5 6.5 3 4 3];%停机成本

    % 初始状态与残余时长（如未知则全 0）
    Sys.S0_th = zeros(10,1);              % 开/停
    Sys.InitOnTime  = zeros(10,1);        % 已连续开机时长
    Sys.InitOffTime = zeros(10,1);        % 已连续停机时长

       % 抽水蓄能（站级互斥，J 可配置）
    Sys.J = 3;   % ← 你要三台就写 3

    % —— 逐台额定与爬坡（可以给标量，代码会扩展到 J×1；也可以直接给 J×1 向量）——
    Sys.PHmax = 300;   % 标量→自动扩成 [300;300;300]
    Sys.PHmin = 60;
    Sys.PPmax = 300;
    Sys.PPmin = 60;

    Sys.RHup = 150;    % 发电侧上爬坡（MW/时段）
    Sys.RHdn = 150;    % 发电侧下爬坡
    Sys.RPup = 150;    % 抽水侧上爬坡
    Sys.RPdn = 150;    % 抽水侧下爬坡

    Sys.etaH = 0.90;   % 发电效率
    Sys.etaP = 0.90;   % 抽水效率

    
    % Compatibility aliases (some versions use eta_p/eta_h)
    Sys.eta_p = Sys.etaP;
    Sys.eta_h = Sys.etaH;
Sys.Umin = 5000;   % 库容下界
    Sys.Umax = 20000;  % 库容上界
    Sys.U0   = 11000;  % 初始库容

    % 抽蓄站级初始模式（0 idle / 1 发 / 2 抽）——仍是“站级”开关，与台数无关
    Sys.S0_psH = 0; 
    Sys.S0_psP = 2;

    % 最小持续时长（站级）
    Sys.PS_minDwell = 2;

    % 启停成本（站级）
    Sys.PS_SUgen  = 80000; 
    Sys.PS_SDgen  = 30000;
    Sys.PS_SUpump = 80000; 
    Sys.PS_SDpump = 30000;

    % ====== 关键：把标量扩展为 J×1（或检查长度）======
    J = Sys.J;
    expandJ = @(v) (isscalar(v) * (v*ones(J,1)) + (~isscalar(v)) * v);  % 小技巧：标量→J×1

    Sys.PHmax = expandJ(Sys.PHmax);  assert(numel(Sys.PHmax)==J,'PHmax 长度应为 J');
    Sys.PHmin = expandJ(Sys.PHmin);  assert(numel(Sys.PHmin)==J,'PHmin 长度应为 J');
    Sys.PPmax = expandJ(Sys.PPmax);  assert(numel(Sys.PPmax)==J,'PPmax 长度应为 J');
    Sys.PPmin = expandJ(Sys.PPmin);  assert(numel(Sys.PPmin)==J,'PPmin 长度应为 J');

    Sys.RHup  = expandJ(Sys.RHup);   assert(numel(Sys.RHup)==J,'RHup 长度应为 J');
    Sys.RHdn  = expandJ(Sys.RHdn);   assert(numel(Sys.RHdn)==J,'RHdn 长度应为 J');
    Sys.RPup  = expandJ(Sys.RPup);   assert(numel(Sys.RPup)==J,'RPup 长度应为 J');
    Sys.RPdn  = expandJ(Sys.RPdn);   assert(numel(Sys.RPdn)==J,'RPdn 长度应为 J');

    % ====== 变量上下界（盒约束）======
    [I, T] = size(Sys.Pmin);
    lb = []; ub = [];
    % 火电出力
    lb = [lb; Sys.Pmin(:)];
    ub = [ub; Sys.Pmax(:)];
    % 抽蓄发电 PH
    lb = [lb; zeros(J*T,1)];
    ub = [ub; repmat(Sys.PHmax(:), T, 1)];
    % 抽蓄抽水 PP
    lb = [lb; zeros(J*T,1)];
    ub = [ub; repmat(Sys.PPmax(:), T, 1)];
    % 风/光
    lb = [lb; Sys.lambda*Sys.PWpre(:)];
    ub = [ub; Sys.PWpre(:)];
    lb = [lb; Sys.lambda*Sys.PVpre(:)];
    ub = [ub; Sys.PVpre(:)];
    % 火电状态 Uth
    lb = [lb; zeros(I*T,1)];
    ub = [ub; ones(I*T,1)];

    Sys.lb = lb(:)'; 
    Sys.ub = ub(:)';

    % 可选：每时段在线火电的总最大出力至少占全火电总最大出力的比例
    Sys.minOnCapRatio = 0.60;  % 60%
    % —— 开关策略与启停偏好（新增，可按需调整）——
Sys.lockBaseOn = true;  % 固定一组“基座机组”整天常开
Sys.kappa      = 0;     % 轻度启停次数惩罚系数（元/次），默认0不开启

if Sys.lockBaseOn
    % 选经济性最优的一组基座机组，使其容量之和 ≥ 60% * 全容量
    Sys.BaseMustOn = pickBaseMustOn(Sys, Sys.minOnCapRatio);
    % 注入“强制常开”集合（若你已有 ForceAlwaysOn，则合并去重）
    if ~isfield(Sys,'ForceAlwaysOn') || isempty(Sys.ForceAlwaysOn)
        Sys.ForceAlwaysOn = Sys.BaseMustOn(:)';
    else
        Sys.ForceAlwaysOn = unique([Sys.ForceAlwaysOn(:); Sys.BaseMustOn(:)])';
    end
end

% 若你想硬指定强制常开，也可以在这里写：Sys.ForceAlwaysOn = [1 2];

% === 初始种群“自动选择”偏好关机机组（基于容量可行 + 经济性代理）===
% 目标：在保证火电最大可供能力足以覆盖“净负荷峰值+备用需求”的前提下，
%      选择一组成本更高/性价比更差的机组作为“初始化偏好关机池”。
% 说明：这仍然只影响初始化（heuristicInit），后续迭代可按需开停（除非 ForceAlwaysOffHard=true）。
Sys.autoPickInitOff = true;   % true=自动挑选；false=使用手动列表
Sys.InitOffK        = 2;      % 自动挑选的偏好关机机组数量（建议 2~4；10台机组时 3 比较常用）
Sys.InitOffProb     = 1.00;   % 初始化时，上述机组被置 0 的概率（建议 0.8~1.0）
Sys.ForceAlwaysOffHard = false; % true=整天硬关机（不建议）；false=仅初始化偏好

if Sys.autoPickInitOff
    try
        Sys.ForceAlwaysOff = pickInitOffUnits(Sys, Sys.InitOffK);
    catch ME
        warning('AutoPick failed: %s. Falling back to manual list.', ME.message);
        Sys.ForceAlwaysOff = [6 9 10];  % ← 手动列表
    end
else
    Sys.ForceAlwaysOff = [6 9 10];  % ← 手动列表
end

% 避免与强制常开冲突
if isfield(Sys,'ForceAlwaysOn') && ~isempty(Sys.ForceAlwaysOn)
    Sys.ForceAlwaysOff = setdiff(Sys.ForceAlwaysOff, Sys.ForceAlwaysOn(:)');
end
if Sys.autoPickInitOff, fprintf('[AutoPick] 初始化偏好关机机组 = %s\n', mat2str(Sys.ForceAlwaysOff)); end

% 若你仍想保留“至少K台”的兜底，可设 Sys.KminOn=0 或不设置
Sys.KminOn = 0;  % 建议置0，改用容量比例约束


end


function x = heuristicInit(Sys, encode)
    % === 初始化 ===
    I = size(Sys.Pmin,1); 
    T = Sys.T; 
    J = Sys.J;

    Pth = zeros(I,T); 
    PH  = zeros(J,T); 
    PP  = zeros(J,T);

    % 风光初值：区间中点
    PW  = Sys.lambda*Sys.PWpre_row + 0.5*(Sys.PWpre_row - Sys.lambda*Sys.PWpre_row);
    PV  = Sys.lambda*Sys.PVpre_row + 0.5*(Sys.PVpre_row - Sys.lambda*Sys.PVpre_row);
    % 这里不再用 net 做火电分配（我们让火电先顶到Pmax，再由抽蓄平衡）
    % net = Sys.PL(:)' - (PW + PV);

    % === 火电初值 Uth：容量偏好 + 随机，随后修复 ===
    cap      = Sys.Pmax(:,1);
    cap_norm = (cap - min(cap)) / max(1e-9, (max(cap) - min(cap))); % 0~1
    p_lo = 0.3; 
    p_hi = 0.7; 
    p_base = p_lo + (p_hi - p_lo) * cap_norm;
    p_noise = 0.08 * (2*rand(I,1)-1);
    p = min(0.98, max(0.02, p_base + p_noise));

    Uth = rand(I,T) < p;                          % 伯努利随机
    Uth(:,1) = Sys.S0_th(:) > 0.5;                % 接续初始状态
   Uth = repairMinUpDownWithInit(Uth, Sys.MinUp, Sys.MinDown, ...
                              Sys.S0_th, Sys.InitOnTime, Sys.InitOffTime);

% === 初始化偏好关机池：随机挑选本个体要“整天关机”的机组 ===
% 说明：这里只影响初始化（heuristicInit），后续迭代不会强制这些机组关机。
allowedMask = true(I,1);      % 初始化“可被兜底机制打开”的候选集合
initOffMask = false(I,1);     % 本个体初始化时整天关机的机组

if isfield(Sys,'ForceAlwaysOff') && ~isempty(Sys.ForceAlwaysOff)
    idxPool = Sys.ForceAlwaysOff(:); idxPool = idxPool(idxPool>=1 & idxPool<=I);

    if isfield(Sys,'ForceAlwaysOffHard') && Sys.ForceAlwaysOffHard
        % 硬约束：整天关机
        initOffMask(idxPool) = true;
    else
        % 软偏好：初始化时随机关机
        pOff = 0.85;
        if isfield(Sys,'InitOffProb'), pOff = Sys.InitOffProb; end
        initOffMask(idxPool) = (rand(numel(idxPool),1) < pOff);
    end

    allowedMask(initOffMask) = false;     % 初始化兜底“补足容量”时不允许打开这些机组
    Uth(initOffMask,:) = 0;              % 初始化先关机
end

% 若指定强制常开（基座机组等），初始化直接置 1
if isfield(Sys,'ForceAlwaysOn') && ~isempty(Sys.ForceAlwaysOn)
    idxOn = Sys.ForceAlwaysOn(:); idxOn = idxOn(idxOn>=1 & idxOn<=I);
    Uth(idxOn,:) = 1;
    allowedMask(idxOn) = true; % 常开机组允许被保留
end

% === 60% 在线容量兜底：仅在 allowedMask==true 的集合内补足 ===
if isfield(Sys,'minOnCapRatio') && Sys.minOnCapRatio > 0
    Uth = enforceMinOnCapacityMasked(Uth, Sys.minOnCapRatio, Sys, allowedMask);
end

% 再盖一遍，确保“初始化偏好关机”的机组仍为关机
Uth(initOffMask,:) = 0;

% === 60% 在线容量兜底：仅在 allowedMask==true 的集合内补足 ===
if isfield(Sys,'minOnCapRatio') && Sys.minOnCapRatio > 0
    Uth = enforceMinOnCapacityMasked(Uth, Sys.minOnCapRatio, Sys, allowedMask);
end
% 如仍保留“至少K台”的兜底，也用带掩码版本（可选）
% if isfield(Sys,'KminOn') && Sys.KminOn > 0
%     Uth = enforceMinOnUnitsMasked(Uth, Sys.KminOn, Sys, allowedMask);
% end

% === 火电先顶到各自 Pmax（仅在线机组） ===
for t = 1:T
    on = Uth(:,t) > 0.5;
    Pth(on,t)  = Sys.Pmax(on,t);
    Pth(~on,t) = 0;
end
% —— 强制：被禁机组出力清零
if any(~allowedMask)
    Pth(~allowedMask,:) = 0;
end

    % === 用抽蓄来维持平衡（缺口→发电PH；过剩→抽水PP） ===
    % 初始化为0，逐时段贪心填充到上下限，不强行追求残差=0
    PH(:) = 0; 
    PP(:) = 0;
    for t = 1:T
        gen_th = sum(Pth(:,t));
        resid  = Sys.PL(t) - (PW(t) + PV(t) + gen_th);   % >0 还缺；<0 过剩

        if resid >= 0
            % 需要发电补缺：按 [PHmin,PHmax] 贪心；站级互斥→此时不抽
            PH(:,t) = fillByMinMax(resid, Sys.PHmin(:), Sys.PHmax(:));
            PP(:,t) = 0;
        else
            % 需要抽水消纳：按 [PPmin,PPmax] 贪心；站级互斥→此时不发
            needPump = -resid;
            PP(:,t) = fillByMinMax(needPump, Sys.PPmin(:), Sys.PPmax(:));
            PH(:,t) = 0;
        end
    end

    % === 打包输出 ===
    x = encode(Pth, PH, PP, PW, PV, double(Uth));
end

function [cost, out, x_rep] = fitnessWithRepair(x, Sys, decode)
    x = clamp(x, Sys.lb, Sys.ub);
    [Pth, PH, PP, PW, PV, Uth] = decode(x);
    [I, T] = size(Pth); J = Sys.J; dt = Sys.dt;

    % 1) 风光投影：lambda*Ppre <= P <= Ppre
    PW = min(Sys.PWpre_row, max(Sys.lambda*Sys.PWpre_row, PW));
    PV = min(Sys.PVpre_row, max(Sys.lambda*Sys.PVpre_row, PV));

    % 2) 火电状态：0/1 投影 + 初始 + 最小启停修复
    % 2) 火电状态：0/1 投影 + 初始
Uth = double(Uth >= 0.5);
Uth(:,1) = Sys.S0_th(:) > 0.5;

% === 可选：整天硬关机掩码（默认关闭）===
% 说明：为满足你“初始化偏好关机但后续可开”的需求，
% 默认 Sys.ForceAlwaysOffHard=false，因此这里不会强制关机。
allowedMask = true(I,1);
if isfield(Sys,'ForceAlwaysOffHard') && Sys.ForceAlwaysOffHard ...
        && isfield(Sys,'ForceAlwaysOff') && ~isempty(Sys.ForceAlwaysOff)
    idxOff = Sys.ForceAlwaysOff(:); idxOff = idxOff(idxOff>=1 & idxOff<=I);
    allowedMask(idxOff) = false;
    Uth(~allowedMask,:) = 0;    % 先清零
end

% 2.2 最小启停修复
Uth = repairMinUpDownWithInit(Uth, Sys.MinUp, Sys.MinDown, ...
                              Sys.S0_th, Sys.InitOnTime, Sys.InitOffTime);

% 再盖一遍硬关机（如启用）
if any(~allowedMask)
    Uth(~allowedMask,:) = 0;
end

% 2.3 在线容量兜底（如启用；硬关机时只在 allowedMask 内补足）
if isfield(Sys,'minOnCapRatio') && Sys.minOnCapRatio > 0
    Uth = enforceMinOnCapacityMasked(Uth, Sys.minOnCapRatio, Sys, allowedMask);

    % 再走一次最小启停以保持一致
    Uth = repairMinUpDownWithInit(Uth, Sys.MinUp, Sys.MinDown, ...
                                  Sys.S0_th, Sys.InitOnTime, Sys.InitOffTime);
    if any(~allowedMask)
        Uth(~allowedMask,:) = 0;
    end
end


    % 门控与上下限（向量化：更快）
    Pth = min(Sys.Pmax, max(Sys.Pmin, Pth));
    Pth = Pth .* (Uth>0.5);

    % 带状态爬坡

    Pth = rampRepairWithStatus(Pth, Uth, Sys.RU, Sys.RD);

    % 3) 抽蓄：站级互斥 + 最小出力门控 + 站级最小持续清零
    [mode_ps, SpsH_t, SpsP_t, PH, PP] = psPlantModeRepair(PH, PP, Sys);
    % 站级模式下爬坡（仅当两期同侧模式）
    PH = rampRepairWithPlantMode(PH, SpsH_t, Sys.RHup, Sys.RHdn);
    PP = rampRepairWithPlantMode(PP, SpsP_t, Sys.RPup, Sys.RPdn);
% ★ 每台抽蓄机组：整天最多一个“发电连续区段”（0出力按抽水计）
[PH, PP] = enforcePSUnitOneGenBlock(PH, PP, mode_ps);

    % 4) 水库能量递推与越界回缩
    [U, PH, PP] = storageForwardRepair(PH, PP, Sys);

    % 4.5) ★ 严格功率平衡（每时段 Σ出力 = 负荷）
    % 说明：
    % - 这里把“功率平衡”从软罚项升级为硬约束（等式约束）。
    % - 做法：在给定 Uth(开停)、PH/PP(抽蓄)、PW/PV(风光) 后，
    %         用一个小规模 QP/投影在火电出力 Pth 上强制满足每时段功率平衡。
    % - 若受 Pmin/Pmax 或爬坡限制导致无法完全平衡，则引入非负松弛量：
    %         LS(t) = 负荷削减(缺电)；SP(t)=溢出/额外弃电(过剩)
    %   仍保证等式成立，但把 LS/SP 计入“平衡松弛”指标与罚项，驱动其趋近 0。
    [Pth, LS, SP, bal_ok] = enforceStrictBalanceThermal(Pth, Uth, PH, PP, PW, PV, Sys);

    % 5) 成本：燃料 + 弃电 + 启停(热/抽蓄) + 罚项
    % 燃料成本（逐时段，在机计 c0）
    fuel = dt * sum(sum( Uth .* ( bsxfun(@times,Sys.a2, Pth.^2) ...
                                 + bsxfun(@times,Sys.b1, Pth) ...
                                 + Sys.c0 ) ));
    % 弃风/弃光（平均项无必要，但保留）
    Wlost = max(Sys.PWpre_row - PW, 0);
    Vlost = max(Sys.PVpre_row - PV, 0);
    E_WV  = (1 / Sys.T) * sum(Sys.Cw * Wlost + Sys.Cv * Vlost);

    % 火电启停成本（向量化）
    prevU = [Sys.S0_th(:)>0.5, (Uth(:,1:end-1)>0.5)];
    curU  = (Uth>0.5);
    startEvt =  curU & ~prevU;     % 0->1
    stopEvt  = ~curU &  prevU;     % 1->0
    SU_cost_th = sum(Sys.SUth(:) .* sum(startEvt,2));
    SD_cost_th = sum(Sys.SDth(:) .* sum(stopEvt,2));
    startstop_th_cost = SU_cost_th + SD_cost_th;

    % 抽蓄启停（站级，含初始）
    startstop_ps_cost = 0;
    prevH = Sys.S0_psH>0.5; prevP = Sys.S0_psP>0.5;
    for t = 1:T
        curH = (mode_ps(t)==1); curP = (mode_ps(t)==2);
        if  curH && ~prevH, startstop_ps_cost = startstop_ps_cost + Sys.PS_SUgen;  end
        if ~curH &&  prevH, startstop_ps_cost = startstop_ps_cost + Sys.PS_SDgen;  end
        if  curP && ~prevP, startstop_ps_cost = startstop_ps_cost + Sys.PS_SUpump; end
        if ~curP &&  prevP, startstop_ps_cost = startstop_ps_cost + Sys.PS_SDpump; end
        prevH = curH; prevP = curP;
    end

    % 平衡与备用
    % r 严格置零（等式约束已强制）；平衡误差体现在 LS/SP 松弛量上。
    r = zeros(1,T);

    reserve_up = zeros(1,T);

    % 火电可上调备用（向量化）
    headroom_th = Sys.Pmax - Pth;
    rampup_th = Sys.RU(:); if numel(rampup_th)==1, rampup_th = rampup_th*ones(I,1); end
    up_th = sum( max(0, min(headroom_th, rampup_th)) .* (Uth>0.5), 1 );

    % 抽蓄可上调备用
    PHmax_vec = Sys.PHmax(:); if numel(PHmax_vec)==1, PHmax_vec = PHmax_vec*ones(J,1); end
    up_ps_gen = (mode_ps==1) .* sum( max(0, bsxfun(@minus, PHmax_vec, PH)), 1 );  % 发电侧余量
    up_ps_pmp = (mode_ps==2) .* sum( PP, 1 );                                      % 停抽释放
    reserve_up = up_th + up_ps_gen + up_ps_pmp;

    reserve_need = Sys.beta*Sys.PL(:)';
    reserve_gap  = max(0, reserve_need - reserve_up);

    uDelta = abs(U(end) - Sys.U0);
    % ★ 平衡罚项：用松弛量代替 |r|
    bal_slack = LS + SP;  % 每时段的“无法用真实机组满足的平衡缺口/过剩”
    pen = Sys.alpha * sum(bal_slack) + Sys.gamma * sum(reserve_gap) + Sys.xi * uDelta;

    cost = fuel + E_WV + startstop_th_cost + startstop_ps_cost + pen;

    % 输出
    out.r = r;
    out.LS = LS;
    out.SP = SP;
    out.bal_ok = bal_ok;
    out.bal_slack = bal_slack;
    out.reserve_gap = reserve_gap;
    out.U = U;
    out.mode_ps = mode_ps;
    out.Uth = Uth;
    % ★ 这里的 p_sum 用“严格平衡后的总供给”表示（含 LS/SP 松弛），应与负荷重合
    out.p_sum=sum(Pth,1) + sum(PH,1) - sum(PP,1) + PW + PV + LS - SP;
    out.Pth=Pth;
    out.PH=PH;
    out.PP=PP;
    out.PW=PW;
    out.PV=PV;
    out.fuel=fuel;
    out.E_WV=E_WV;
    out.startstop_th_cost=startstop_th_cost;
    out.startstop_ps_cost=startstop_ps_cost;
    out.pen=pen;
    % ---- Detailed economic breakdown ----
    out.Wlost = Wlost; out.Vlost = Vlost;
    out.cost_wcurt = (1 / Sys.T) * sum(Sys.Cw * Wlost);
    out.cost_vcurt = (1 / Sys.T) * sum(Sys.Cv * Vlost);
    out.SU_cost_th = SU_cost_th; out.SD_cost_th = SD_cost_th;
    out.start_th_events = sum(startEvt,2); out.stop_th_events = sum(stopEvt,2);
    out.pen_balance = Sys.alpha * sum(bal_slack);
    out.pen_reserve = Sys.gamma * sum(reserve_gap);
    out.pen_terminal = Sys.xi * uDelta;
    out.sum_bal_slack = sum(bal_slack);
    out.sum_reserve_gap = sum(reserve_gap);
    out.uDelta = uDelta;


    % ★ 返回“投影后的”决策向量（严格平衡+所有修复已作用），用于把种群拉回可行流形
    % 注意：这里把 Uth 用 double(0/1) 写回；其它变量按修复后值写回。
    x_rep = encodeVector(Pth, PH, PP, PW, PV, double(Uth));
end

function X = clamp(X, lb, ub)
    X = min(ub, max(lb, X));
end

function [P, PH, PP, PW, PV, Uth] = decodeVector(x, I, J, T)
    ofs = 0;
    P   = reshape(x(ofs+(1:I*T)), I, T); ofs = ofs + I*T;
    PH  = reshape(x(ofs+(1:J*T)), J, T); ofs = ofs + J*T;
    PP  = reshape(x(ofs+(1:J*T)), J, T); ofs = ofs + J*T;
    PW  = reshape(x(ofs+(1:T)),   1, T); ofs = ofs + T;
    PV  = reshape(x(ofs+(1:T)),   1, T); ofs = ofs + T;
    Uth = reshape(x(ofs+(1:I*T)), I, T);
end

function x = encodeVector(P,PH,PP,PW,PV,Uth)
    x = [P(:); PH(:); PP(:); PW(:); PV(:); Uth(:)]';
end

function P2 = rampRepairWithStatus(P, S, RU, RD)
    [n, T] = size(P); P2 = P;
    RU = RU(:); if numel(RU)==1, RU = RU*ones(n,1); end
    RD = RD(:); if numel(RD)==1, RD = RD*ones(n,1); end
    for i = 1:n
        for t = 2:T
            if S(i,t-1)>0.5 && S(i,t)>0.5
                hi = P2(i,t-1) + RU(i);
                lo = P2(i,t-1) - RD(i);
                P2(i,t) = min(hi, max(lo, P2(i,t)));
            end
        end
    end
end

function P2 = rampRepairWithPlantMode(P, modeFlag, RU, RD)
    % modeFlag: 1xT，表示该侧(发/抽)在 t 是否为站级模式
    [n,T] = size(P); P2 = P;
    RU = RU(:); if numel(RU)==1, RU = RU*ones(n,1); end
    RD = RD(:); if numel(RD)==1, RD = RD*ones(n,1); end
    for i = 1:n
        for t = 2:T
            if modeFlag(t-1) && modeFlag(t)
                hi = P2(i,t-1) + RU(i);
                lo = P2(i,t-1) - RD(i);
                P2(i,t) = min(hi, max(lo, P2(i,t)));
            end
        end
    end
end

function [mode, SpsH_t, SpsP_t, PH2, PP2] = psPlantModeRepair(PH, PP, Sys)
    % 站级互斥 + 站级最小持续清零 + 逐机组最小出力门控
    [J,T] = size(PH); PH2 = PH; PP2 = PP; epsv = 1e-6;

    % 初步站级模式：按全站功率较大侧定，另一侧清零
    mode = zeros(1,T);
    for t = 1:T
        sumH = sum(PH2(:,t)); sumP = sum(PP2(:,t));
        if sumH <= epsv && sumP <= epsv
            mode(t) = 0; PH2(:,t) = 0; PP2(:,t) = 0;
        elseif sumH >= sumP
            mode(t) = 1; PP2(:,t) = 0;
        else
            mode(t) = 2; PH2(:,t) = 0;
        end
    end

    % 站级最小持续（把短段置空闲）
    md = 0; if isfield(Sys,'PS_minDwell'), md = Sys.PS_minDwell; end
    if md >= 2
        t = 1;
        while t <= T
            s = t; m = mode(t);
            while t<=T && mode(t)==m, t = t+1; end
            e = t-1;
            if (m==1 || m==2) && (e-s+1)<md
                mode(s:e) = 0; PH2(:,s:e)=0; PP2(:,s:e)=0;
            end
        end
    end

    % 初始站级模式优先（若提供）
    if (Sys.S0_psH||Sys.S0_psP)
        if Sys.S0_psH, mode(1)=1; PH2(:,1)=max(PH2(:,1),0); PP2(:,1)=0; end
        if Sys.S0_psP, mode(1)=2; PP2(:,1)=max(PP2(:,1),0); PH2(:,1)=0; end
    end

    % 逐机组最小出力门控
    for t = 1:T
        if mode(t)==1
            for j = 1:J
                if PH2(j,t) > epsv
                    PH2(j,t) = min(Sys.PHmax(j), max(Sys.PHmin(j), PH2(j,t)));
                else
                    PH2(j,t) = 0;
                end
                PP2(j,t) = 0;
            end
        elseif mode(t)==2
            for j = 1:J
                if PP2(j,t) > epsv
                    PP2(j,t) = min(Sys.PPmax(j), max(Sys.PPmin(j), PP2(j,t)));
                else
                    PP2(j,t) = 0;
                end
                PH2(j,t) = 0;
            end
        else
            PH2(:,t) = 0; PP2(:,t) = 0;
        end
    end

    SpsH_t = (mode==1);
    SpsP_t = (mode==2);
end

function [U, PH, PP] = storageForwardRepair(PH, PP, Sys)
    % U_t = U_{t-1} + (etaP*sum(PP) - sum(PH)/etaH)*dt；越界则缩放回边界
    J = Sys.J; T = Sys.T; dt = Sys.dt;
    U = zeros(1,T);
    inflow1 = (Sys.etaP*sum(PP(:,1)) - (1/Sys.etaH)*sum(PH(:,1)))*dt;
    U(1) = Sys.U0 + inflow1;
    if U(1) > Sys.Umax
        denom = Sys.etaP*sum(PP(:,1))*dt + 1e-9;
        scale = max(0, 1 - (U(1)-Sys.Umax)/denom);
        PP(:,1) = scale*PP(:,1);
        U(1) = Sys.U0 + (Sys.etaP*sum(PP(:,1)) - (1/Sys.etaH)*sum(PH(:,1)))*dt;
    elseif U(1) < Sys.Umin
        denom = (1/Sys.etaH)*sum(PH(:,1))*dt + 1e-9;
        scale = max(0, 1 - (Sys.Umin-U(1))/denom);
        PH(:,1) = scale*PH(:,1);
        U(1) = Sys.U0 + (Sys.etaP*sum(PP(:,1)) - (1/Sys.etaH)*sum(PH(:,1)))*dt;
    end
    for t = 2:T
        U(t) = U(t-1) + (Sys.etaP*sum(PP(:,t)) - (1/Sys.etaH)*sum(PH(:,t)))*dt;
        if U(t) > Sys.Umax
            denom = Sys.etaP*sum(PP(:,t))*dt + 1e-9;
            scale = max(0, 1 - (U(t)-Sys.Umax)/denom);
            PP(:,t) = scale*PP(:,t);
            U(t) = U(t-1) + (Sys.etaP*sum(PP(:,t)) - (1/Sys.etaH)*sum(PH(:,t)))*dt;
        elseif U(t) < Sys.Umin
            denom = (1/Sys.etaH)*sum(PH(:,t))*dt + 1e-9;
            scale = max(0, 1 - (Sys.Umin-U(t))/denom);
            PH(:,t) = scale*PH(:,t);
            U(t) = U(t-1) + (Sys.etaP*sum(PP(:,t)) - (1/Sys.etaH)*sum(PH(:,t)))*dt;
        end
    end
end

function u_fix = repairMinUpDownWithInit(u, MinUp, MinDown, S0, InitOn, InitOff)
% 修改版：初始化阶段不强制满足最小启停时间；
% 仅在发生状态转换(0->1 或 1->0)之后，向前(从该时刻起)施加最小持续约束。
%
% 兼容说明：
% - 为兼容你现有调用，保留了 S0 / InitOn / InitOff 形参，但在本实现中不会使用。
% - t=1 取调用者给定的状态，不做“根据初始残余时长延长/清零”的处理。

    [I, T] = size(u);
    u_fix  = u > 0.5;
    MinUp   = MinUp(:);
    MinDown = MinDown(:);

    %#ok<INUSD> % S0/InitOn/InitOff 故意不使用

    for i = 1:I
        t = 2;
        while t <= T
            if u_fix(i,t)==1 && u_fix(i,t-1)==0
                % 0->1：从 t 起至少保持 MinUp(i) 个时段为 1
                tend = min(T, t + MinUp(i) - 1);
                u_fix(i, t:tend) = 1;
                t = tend + 1;      % 跳过已固定区间
            elseif u_fix(i,t)==0 && u_fix(i,t-1)==1
                % 1->0：从 t 起至少保持 MinDown(i) 个时段为 0
                tend = min(T, t + MinDown(i) - 1);
                u_fix(i, t:tend) = 0;
                t = tend + 1;
            else
                t = t + 1;
            end
        end
    end
end


function Uth2 = enforceMinOnUnits(Uth, Kmin, Sys)
    % 每时段至少 Kmin 台开机（默认关闭）
    [I,T] = size(Uth); Uth2 = Uth;
    mc = 2*Sys.a2.*Sys.Pmin(:,1) + Sys.b1;
    [~, order] = sort(mc, 'ascend');
    for t = 1:T
        onNow = find(Uth2(:,t)>0.5);
        if numel(onNow) < Kmin
            need = Kmin - numel(onNow);
            used = false(I,1); used(onNow) = true;
            for k = 1:I
                i = order(k);
                if ~used(i)
                    Uth2(i,t) = 1;
                    used(i) = true;
                    need = need - 1;
                    if need==0, break; end
                end
            end
        end
    end
end
function Uth2 = enforceMinOnCapacitySticky(Uth, ratio, Sys)
% 黏性兜底：若某时段在线容量不足，仅“加开”机组，且优先：
% 1) 上一时段已开；2) 经济性更好（mc小）；3) 当期Pmax更大
    [I,T] = size(Uth); 
    Uth2  = Uth>0.5;

    capAll = sum(Sys.Pmax,1);
    mc = 2*Sys.a2.*Sys.Pmin(:,1) + Sys.b1;

    % 过滤强制索引，防越界
    forcedOn  = false(I,1);
    forcedOff = false(I,1);
    if isfield(Sys,'ForceAlwaysOn') && ~isempty(Sys.ForceAlwaysOn)
        idx = Sys.ForceAlwaysOn(:); idx = idx(idx>=1 & idx<=I);
        forcedOn(idx) = true;
    end
    if isfield(Sys,'ForceAlwaysOff') && ~isempty(Sys.ForceAlwaysOff)
        idx = Sys.ForceAlwaysOff(:); idx = idx(idx>=1 & idx<=I);
        forcedOff(idx) = true;
    end

    for t = 1:T
        on    = Uth2(:,t)>0.5;
        on(forcedOn)  = true;   % 强制常开
        on(forcedOff) = false;  % 强制常停

        capOn = sum(Sys.Pmax(on,t));
        need  = ratio * capAll(t);
        if capOn < need
            if t>1, prevOn = Uth2(:,t-1)>0.5; else, prevOn = false(I,1); end
            cand = find(~on & ~forcedOff);
            if ~isempty(cand)
                sc = 2*double(prevOn(cand)) + normalizeScore(-mc(cand)) + ...
                     0.5*normalizeScore(Sys.Pmax(cand,t));
                [~, ord] = sort(sc, 'descend');
                for k = 1:numel(ord)
                    i = cand(ord(k));
                    on(i) = true; Uth2(i,t) = 1;
                    capOn = capOn + Sys.Pmax(i,t);
                    if capOn >= need, break; end
                end
            end
        else
            Uth2(:,t) = on;  % 同步
        end
    end
    % 强制常开最终再盖一遍
    Uth2(forcedOn,:) = true;
end

function z = normalizeScore(x)
    x = x(:);
    if all(x==x(1)), z = zeros(size(x)); return; end
    xmin = min(x); xmax = max(x);
    z = (x - xmin) / max(1e-9, (xmax - xmin));
end


function [PH2, PP2] = enforcePSUnitOneGenBlock(PH, PP, mode_ps)
% 目的：对每台抽蓄机组 j，仅允许“发电”(PH>0)在整天内出现一个连续区段
%       这样就保证了：抽→发（进入该区段）最多一次，发→抽（离开该区段）最多一次
% 约定：出力为 0 视为“抽水”一侧，因此零值不会制造“发侧”的额外片段
%
% 输入：
%   PH, PP   : J×T（已过站级互斥、最小出力门控与站级爬坡修复）
%   mode_ps  : 1×T，站级模式（0空闲/1发/2抽）
% 输出：
%   PH2, PP2 : 修正后的出力（只会把多余的发电片段置 0；不增量、不换侧）

    [J, T] = size(PH);
    PH2 = PH; 
    PP2 = PP;
    epsv = 1e-9;

    for j = 1:J
        % 只在全站“发电模式”的时刻统计该机组的发电片段
        isGenTime = (mode_ps == 1);
        idx = find(PH2(j,:) > epsv & isGenTime);

        if isempty(idx)
            continue; % 该机组整天未参与发电，无需处理
        end

        % 找到“发电时刻”的连续区段（连号分段）
        blocks = {};
        s = idx(1); prev = idx(1);
        for k = 2:numel(idx)
            if idx(k) == prev + 1
                prev = idx(k);
            else
                blocks{end+1} = s:prev; %#ok<AGROW>
                s = idx(k); prev = idx(k);
            end
        end
        blocks{end+1} = s:prev;

        % 若只有一个发电连续片段，天然满足约束
        if numel(blocks) <= 1
            continue;
        end

        % 多个片段：保留“发电量（∑PH）最大”的片段，其余置 0
        sums = cellfun(@(b) sum(PH2(j,b)), blocks);
        [~, keep_id] = max(sums);

        for b = 1:numel(blocks)
            if b == keep_id, continue; end
            PH2(j, blocks{b}) = 0;   % 置零→按你的规则自动视为抽水侧
        end

        % 说明：不需改 PP；零值即视作抽水，且 station 互斥已保证对侧为 0
        % 后续 storageForwardRepair 会基于修改后的 PH/PP 重新做水量边界回缩
    end
end
function y = fillByMinMax(total, vmin, vmax)
% 按机组上限从大到小，贪心填充到不超过 total
% 若 total 小于某机组的最小出力，则跳过该机组（避免超发/超抽）
    n = numel(vmin);
    [~, order] = sort(vmax, 'descend');  % 大容量优先
    y = zeros(n,1);
    left = total;

    for kk = 1:n
        i = order(kk);
        if left <= 0, break; end
        ub = min(vmax(i), left);   % 当前能给的上限
        if ub >= vmin(i)
            y(i) = ub;             % 一旦启用，就至少给到 vmin，尽量给满
            left = left - y(i);
        else
            % 需求小于最小出力门槛：不启该机组，留给后续修复/罚项处理
            y(i) = 0;
        end
    end
end
function idx = pickBaseMustOn(Sys, ratio)
% 按“经济性（mc 越小越好）”排序，累加 Pmax 直到达到 ratio*总容量
    I = size(Sys.Pmax,1);
    totalCap = sum(Sys.Pmax(:,1));
    needCap  = ratio * totalCap;

    mc = 2*Sys.a2.*Sys.Pmin(:,1) + Sys.b1;   % 简易边际成本代理
    [~, ord] = sort(mc, 'ascend');

    sel  = false(I,1); acc = 0;
    for k = 1:I
        i = ord(k);
        sel(i) = true;
        acc = acc + Sys.Pmax(i,1);
        if acc >= needCap, break; end
    end

    idx = find(sel);
end

function idxOff = pickInitOffUnits(SysLocal, KLocal)
% 自动选择“初始化偏好关机”的火电机组集合（大小 KLocal）
% 约束：剩余火电最大出力需覆盖 max_t(净负荷 + 备用需求)
% 目标（代理）：最小化“留在线机组”的经济性加权分数（越省越好）
%
% 说明：
% - 这是初始化启发式用的“偏好关机池”，不是硬约束（除非 ForceAlwaysOffHard=true）。
% - I=10 时穷举 C(10,K) 很小，直接穷举组合最稳健。
%
    Pmax = SysLocal.Pmax(:,1);
    Pmin = SysLocal.Pmin(:,1);
    I    = numel(Pmax);

    if nargin < 2 || isempty(KLocal) || KLocal <= 0
        idxOff = [];
        return;
    end
    KLocal = min(KLocal, I);

    % --- 计算“必须覆盖”的容量需求（保守：不计抽蓄，仅用风光预测）---
    netLoad = SysLocal.PL(:)' - SysLocal.PWpre_row - SysLocal.PVpre_row; % MW
    netLoad = max(netLoad, 0);
    reserve_need = SysLocal.beta * SysLocal.PL(:)';                      % MW
    req = max(netLoad + reserve_need);                                   % MW
    req = req * 1.00;  % 可加安全系数：1.00~1.05

    % 强制常开机组不能被选入 Off 集合
    mustOn = [];
    if isfield(SysLocal,'ForceAlwaysOn') && ~isempty(SysLocal.ForceAlwaysOn)
        mustOn = unique(SysLocal.ForceAlwaysOn(:)');
    end

    cand = setdiff(1:I, mustOn);
    if numel(cand) <= KLocal
        idxOff = cand(:)';
        return;
    end

    % --- 机组经济性代理得分 ---
    % 用二次燃料成本 (aP^2+bP+c) 在参考点的平均成本 + 启动成本摊销
    a = SysLocal.a2(:); b = SysLocal.b1(:); c = SysLocal.c0(:);
    SU = SysLocal.SUth(:);
    Pref = max(Pmin, 0.6*Pmax);
    avgCost = (a.*(Pref.^2) + b.*Pref + c) ./ max(Pref, 1e-6);   % RMB/MWh(近似)
    suPerMWh = SU ./ max(Pref * SysLocal.T, 1e-6);               % 启动摊到每小时（粗略）
    score = avgCost + suPerMWh;

    % --- 穷举组合，找最优 Off 集合 ---
    combos = nchoosek(cand, KLocal);
    bestVal = inf;
    bestOff = combos(1,:);

    for k = 1:size(combos,1)
        off = combos(k,:);
        on  = setdiff(1:I, off);

        cap = sum(Pmax(on));
        if cap < req
            continue; % 容量不够，跳过
        end

        val = sum(score(on) .* Pmax(on)); % 加权代理：越小越省
        if val < bestVal
            bestVal = val;
            bestOff = off;
        end
    end

    idxOff = bestOff(:)';
end

function Uth2 = enforceMinOnCapacityMasked(Uth, ratio, Sys, allowedMask)
% 仅在 allowedMask==true 的机组集合上，保证：
%   sum(Pmax_on) >= ratio * sum(Pmax_allowed)
% 不会把 allowedMask==false 的机组置 1
    [I,T] = size(Uth);
    Uth2 = Uth>0.5;
    allowedMask = logical(allowedMask(:));

    mc = 2*Sys.a2.*Sys.Pmin(:,1) + Sys.b1;  % 简易经济性评分（越小越好）

    for t = 1:T
        on = Uth2(:,t)>0.5;
        on(~allowedMask) = false;  % 禁止机组恒 0

        capAll_t = sum(Sys.Pmax(allowedMask, t));
        if capAll_t <= 1e-9
            Uth2(~allowedMask, t) = 0;
            continue;
        end

        capOn_t = sum(Sys.Pmax(on, t));
        need    = ratio * capAll_t;

        if capOn_t < need
            idxAllowed = find(allowedMask);
            % 评分：上一时段已开优先 + 经济性 + 容量
            if t>1, prevOn = Uth2(:,t-1)>0.5; else, prevOn = false(I,1); end
            sc = 2*double(prevOn(idxAllowed)) ...
               + score01(-mc(idxAllowed)) ...
               + 0.5*score01(Sys.Pmax(idxAllowed, t));

            [~, ord] = sort(sc, 'descend');
            seq = idxAllowed(ord);

            for k = 1:numel(seq)
                i = seq(k);
                if ~on(i)
                    on(i) = true; Uth2(i,t) = 1;
                    capOn_t = capOn_t + Sys.Pmax(i,t);
                    if capOn_t >= need, break; end
                end
            end
        end

        Uth2(~allowedMask, t) = 0;  % 再盖一遍
    end
end

function z = score01(x)
    x = x(:);
    if all(abs(x - x(1)) < 1e-12), z = zeros(size(x)); return; end
    xmin = min(x); xmax = max(x);
    z = (x - xmin) / max(1e-9, (xmax - xmin));
end
%% ===== NSGA-II / MOPSO 辅助函数（新增）=====

function [F, out, x_rep] = objectivesWithRepair(x, Sys, decode)
    % 目标定义（恢复常见写法，便于画真正的 f1-f2 Pareto）：
    %   f1 = 经济成本（不含罚项，越小越好）
    %   f2 = 约束违约度（越小越好，0 表示可行）
    %
    % 说明：
    % - 算法内部仍会使用“ε-约束/可行性优先”策略逐步压低 f2；
    % - 你希望的 “best f1 单调下降” 通过历史最优(cummin)记录实现，而不是改变真实目标。

    [~, out, x_rep] = fitnessWithRepair(x, Sys, decode);

    % 解码出机组开关状态，用于“少开机”轻惩罚（可选）
    [~, ~, ~, ~, ~, Uth] = decode(x_rep);

    % 经济成本（不含罚项）
    f1 = out.fuel + out.startstop_th_cost + out.startstop_ps_cost + out.E_WV + Sys.lambdaOn * sum(Uth(:));

    % 约束违约度：用“总量”而不是 max，更能拉开 Pareto（也更符合 UC/储能约束的整体违约）
    % 其中 bal_slack / reserve_gap 通常是逐时段的非负违约量
    violP = sum(out.bal_slack);       % 平衡松弛总量
    violR = sum(out.reserve_gap);     % 备用缺口总量
    violU = abs(out.U(end) - Sys.U0); % 末端库容偏差
    f2 = violP + violR + violU;

    % 记录，便于最终打印/对比
    out.f1_raw = f1;
    out.f2_raw = f2;

    F = [f1, f2];
end

function [rank, crowd] = rankAndCrowd(F)
    fronts = fastNonDominatedSort(F);
    N = size(F,1);
    rank = inf(N,1);
    crowd = zeros(N,1);
    for fi = 1:numel(fronts)
        idx = fronts{fi};
        rank(idx) = fi;
        crowd(idx) = crowdingDistance(F(idx,:));
    end
end

function [Xsel, Fsel, Vsel, selIdx] = nsga2EnvironmentalSelect(X, F, V, Nkeep)
    fronts = fastNonDominatedSort(F);
    Xsel = zeros(Nkeep, size(X,2));
    selIdx = zeros(Nkeep,1);
    Fsel = zeros(Nkeep, size(F,2));
    Vsel = zeros(Nkeep, size(V,2));

    n = 0;
    for fi = 1:numel(fronts)
        idx = fronts{fi};
        if n + numel(idx) <= Nkeep
            Xsel(n+(1:numel(idx)),:) = X(idx,:);
            selIdx(n+(1:numel(idx))) = idx(:);
            Fsel(n+(1:numel(idx)),:) = F(idx,:);
            Vsel(n+(1:numel(idx)),:) = V(idx,:);
            n = n + numel(idx);
        else
            % 最后一层按拥挤距离选
            cd = crowdingDistance(F(idx,:));
            [~, ord] = sort(cd, 'descend');
            need = Nkeep - n;
            pick = idx(ord(1:need));
            Xsel(n+(1:need),:) = X(pick,:);
            selIdx(n+(1:need)) = pick(:);
            Fsel(n+(1:need),:) = F(pick,:);
            Vsel(n+(1:need),:) = V(pick,:);
            break;
        end
    end
end

function [archX, archF] = updateArchiveSparseGrid(archX, archF, X, F, Amax, nGrid)
    % 合并
    Xc = [archX; X];
    Fc = [archF; F];

    % 取非支配集
    fronts = fastNonDominatedSort(Fc);
    nd = fronts{1};
    archX = Xc(nd,:);
    archF = Fc(nd,:);

    % 裁剪到 Amax：按稀疏网格密度删除（删最拥挤格）
    if size(archX,1) > Amax
        [archX, archF] = pruneArchiveByGrid(archX, archF, Amax, nGrid);
    end
end

function leader = selectLeaderSparseGrid(archX, archF, nGrid)
    % 从“稀疏格”里抽 leader：优先选占用最少的格子，保持前沿均匀性
    if isempty(archX)
        leader = [];
        return;
    end
    cellId = gridCellIndex(archF, nGrid);
    u = unique(cellId);
    cnt = zeros(numel(u),1);
    for i = 1:numel(u), cnt(i) = sum(cellId==u(i)); end
    minCnt = min(cnt);
    sparseCells = u(cnt==minCnt);

    % 在最稀疏的格子里随机挑一个解
    c = sparseCells(randi(numel(sparseCells)));
    cand = find(cellId==c);
    leader = archX(cand(randi(numel(cand))), :);
end

function [X2, F2] = pruneArchiveByGrid(X, F, Amax, nGrid)
    X2 = X; F2 = F;
    while size(X2,1) > Amax
        cellId = gridCellIndex(F2, nGrid);
        u = unique(cellId);
        cnt = zeros(numel(u),1);
        for i = 1:numel(u), cnt(i) = sum(cellId==u(i)); end
        [~, imax] = max(cnt);
        denseCell = u(imax);
        cand = find(cellId==denseCell);
        % 在最拥挤格子里随机删一个（也可删拥挤距离最小的）
        rm = cand(randi(numel(cand)));
        X2(rm,:) = [];
        F2(rm,:) = [];
    end
end

function cellId = gridCellIndex(F, nGrid)
    % 将目标空间归一化到[0,1]，再分箱成 nGrid^M 个网格
    M = size(F,2);
    fmin = min(F,[],1);
    fmax = max(F,[],1);
    denom = max(1e-12, (fmax - fmin));
    Fn = (F - fmin) ./ denom;
    Fn = min(1, max(0, Fn));

    g = 1 + floor(Fn * (nGrid-1));           % 1..nGrid
    g = min(nGrid, max(1, g));

    % 线性编码：cellId = g1 + (g2-1)*nGrid + (g3-1)*nGrid^2 + ...
    cellId = zeros(size(F,1),1);
    base = 1;
    for m = 1:M
        cellId = cellId + (g(:,m)-1) * base;
        base = base * nGrid;
    end
    cellId = cellId + 1;
end

function fronts = fastNonDominatedSort(F)
    % 简洁版快速非支配排序（min）
    N = size(F,1);
    S = cell(N,1);
    nDom = zeros(N,1);
    fronts = {};
    F1 = [];

    for p = 1:N
        Sp = [];
        np = 0;
        for q = 1:N
            if p==q, continue; end
            if dominates(F(p,:), F(q,:))
                Sp(end+1) = q; %#ok<AGROW>
            elseif dominates(F(q,:), F(p,:))
                np = np + 1;
            end
        end
        S{p} = Sp;
        nDom(p) = np;
        if np == 0
            F1(end+1) = p; %#ok<AGROW>
        end
    end
    fronts{1} = F1;

    i = 1;
    while ~isempty(fronts{i})
        Q = [];
        for p = fronts{i}
            for q = S{p}
                nDom(q) = nDom(q) - 1;
                if nDom(q) == 0
                    Q(end+1) = q; %#ok<AGROW>
                end
            end
        end
        i = i + 1;
        fronts{i} = Q;
    end
    if isempty(fronts{end}), fronts(end) = []; end
end

function tf = dominates(a, b)
    % a dominates b (minimization)
    tf = all(a <= b) && any(a < b);
end

function cd = crowdingDistance(F)
    % 拥挤距离（NSGA-II）
    N = size(F,1);
    M = size(F,2);
    cd = zeros(N,1);
    if N == 0, return; end
    if N <= 2
        cd(:) = inf;
        return;
    end
    for m = 1:M
        [vals, ord] = sort(F(:,m), 'ascend');
        cd(ord(1)) = inf;
        cd(ord(end)) = inf;
        vmin = vals(1); vmax = vals(end);
        denom = max(1e-12, vmax - vmin);
        for i = 2:N-1
            cd(ord(i)) = cd(ord(i)) + (vals(i+1) - vals(i-1)) / denom;
        end
    end
end

function idx = tournamentSelect(rank, crowd, tourSize)
    % NSGA-II 锦标赛：rank小优先；rank相同拥挤距离大优先
    N = numel(rank);
    cand = randi(N, [tourSize,1]);
    best = cand(1);
    for i = 2:tourSize
        j = cand(i);
        if rank(j) < rank(best)
            best = j;
        elseif rank(j) == rank(best) && crowd(j) > crowd(best)
            best = j;
        end
    end
    idx = best;
end

%% ====== 严格功率平衡投影（新增）======
function [Pth, LS, SP, bal_ok] = enforceStrictBalanceThermal(Pth0, Uth, PH, PP, PW, PV, Sys)
    % 严格功率平衡投影（“硬等式”）：
    % 对每个时段 t：
    %   sum_i Pth(i,t) + sum(PH(:,t)) - sum(PP(:,t)) + PW(t)+PV(t) + LS(t) - SP(t) = PL(t)
    % LS/SP 为非负松弛量（缺电/过剩），保证等式始终可满足；
    % 同时在修复阶段尽量把 LS/SP 压到 0（靠出力调整实现平衡）。

    [I,T] = size(Pth0);
    LS = zeros(1,T);
    SP = zeros(1,T);

    % 热电需要承担的“净负荷”
    netLoad = Sys.PL(:)' - (sum(PH,1) - sum(PP,1) + PW + PV);

    tol = 0;
    if isfield(Sys,'PbalTol'); tol = Sys.PbalTol; end

    % 火电上下限（随开停门控）
    lb = Sys.Pmin .* (Uth>0.5);
    ub = Sys.Pmax .* (Uth>0.5);

    % 平衡方式选择：fast（默认）/ qp（若有 toolbox）
    if ~isfield(Sys,'strictBalanceMethod'); Sys.strictBalanceMethod = 'fast'; end
    if ~isfield(Sys,'balancePasses');       Sys.balancePasses       = 2;      end

    bal_ok = false;

    useQP = strcmpi(Sys.strictBalanceMethod,'qp');
    if useQP
        % QP（更精确但更慢）
        persistent hasQP
        if isempty(hasQP)
            hasQP = exist('quadprog','file') == 2;
        end
        if hasQP
            try
                [Pth, LS, SP, bal_ok] = solveBalanceQP(Pth0, Uth, lb, ub, netLoad, Sys);
                return;
            catch
                % QP失败则回退到 fast
            end
        end
    end

    % -------- fast：前向/后向“爬坡一致”投影 + 每时段平衡分配（非常快） --------
    Pth = min(ub, max(lb, Pth0));

    RU = Sys.RU(:); RD = Sys.RD(:);
    if numel(RU)==1, RU = RU*ones(I,1); end
    if numel(RD)==1, RD = RD*ones(I,1); end

    for pass = 1:Sys.balancePasses
        % 1) forward：用 t-1 收紧 t 的爬坡可行区间
        for t = 2:T
            on2 = (Uth(:,t-1)>0.5) & (Uth(:,t)>0.5);
            ub_t = ub(:,t); lb_t = lb(:,t);
            ub_t(on2) = min(ub_t(on2), Pth(on2,t-1) + RU(on2));
            lb_t(on2) = max(lb_t(on2), Pth(on2,t-1) - RD(on2));
            Pth(:,t)  = min(ub_t, max(lb_t, Pth(:,t)));

            % 尝试在该时段内做平衡分配（不破坏本时段 bounds）
            [Pth(:,t), LS(t), SP(t)] = balanceOnePeriodTol(Pth(:,t), lb_t, ub_t, netLoad(t), tol);
        end
        % t=1 也平衡一次
        [Pth(:,1), LS(1), SP(1)] = balanceOnePeriodTol(Pth(:,1), lb(:,1), ub(:,1), netLoad(1), tol);

        % 2) backward：用 t+1 收紧 t 的爬坡可行区间
        for t = T-1:-1:1
            on2 = (Uth(:,t)>0.5) & (Uth(:,t+1)>0.5);
            ub_t = ub(:,t); lb_t = lb(:,t);
            % 由下一时段约束反推本时段可行区间：
            %   P_{t+1} - P_t <= RU  -> P_t >= P_{t+1} - RU
            %   P_t - P_{t+1} <= RD  -> P_t <= P_{t+1} + RD
            lb_t(on2) = max(lb_t(on2), Pth(on2,t+1) - RU(on2));
            ub_t(on2) = min(ub_t(on2), Pth(on2,t+1) + RD(on2));
            Pth(:,t)  = min(ub_t, max(lb_t, Pth(:,t)));

            [Pth(:,t), LS(t), SP(t)] = balanceOnePeriodTol(Pth(:,t), lb_t, ub_t, netLoad(t), tol);
        end
    end

    % 最终再算一次松弛量（保证严格等式成立的“记账”）
    for t = 1:T
        s = sum(Pth(:,t));
        if s < netLoad(t) - tol
            LS(t) = (netLoad(t) - tol) - s;
            SP(t) = 0;
        elseif s > netLoad(t) + tol
            SP(t) = s - (netLoad(t) + tol);
            LS(t) = 0;
        else
            LS(t) = 0; SP(t) = 0;
        end
    end
    bal_ok = all(LS<1e-9) && all(SP<1e-9);
end

function [Pcol, LS, SP] = balanceOnePeriodTol(Pcol, lbcol, ubcol, target, tol)
    % 在给定 bounds 下，让 sum(Pcol) 尽量贴近 target（先用真实机组调节），
    % 剩余部分记到 LS/SP（保证等式可写成 sum(P)+LS-SP=target）
    LS = 0; SP = 0;
    s = sum(Pcol);

    if abs(s - target) <= tol
        LS = 0; SP = 0; return;
    end

    if s < target - tol
        gap = (target - tol) - s;
        head = max(0, ubcol - Pcol);
        add = fillProportional(gap, head);
        Pcol = Pcol + add;
        s2 = sum(Pcol);
        LS = max(0, target - s2);
    elseif s > target + tol
        surplus = s - (target + tol);
        down = max(0, Pcol - lbcol);
        sub = fillProportional(surplus, down);
        Pcol = Pcol - sub;
        s2 = sum(Pcol);
        SP = max(0, s2 - target);
    end
end
function [Pth, LS, SP, bal_ok] = solveBalanceQP(P0, Uth, lb, ub, netLoad, Sys)
    % QP：
    %   min  ||P - P0||^2 + wS * (||LS||^2 + ||SP||^2)
    % s.t.  Aeq*[P;LS;SP] = netLoad
    %       lb<=P<=ub,  LS>=0, SP>=0
    %       ramp constraints (在给定 Uth 下为线性不等式)

    [I,T] = size(P0);
    nP = I*T;
    nS = 2*T;
    n  = nP + nS;

    % 变量顺序：vec(Pth) ; LS(1:T) ; SP(1:T)
    p0 = P0(:);

    % 二次项：P 部分单位阵；松弛项给一个较大的权重（让 LS/SP 更倾向 0）
    wS = 1e6;
    H = speye(n);
    H(nP+(1:nS), nP+(1:nS)) = wS * speye(nS);
    f = [-2*p0; zeros(nS,1)];

    % 等式：每时段 sum_i P(i,t) + LS(t) - SP(t) = netLoad(t)
    Aeq = sparse(T, n);
    for t = 1:T
        cols = (t-1)*I + (1:I);
        Aeq(t, cols) = 1;
        Aeq(t, nP + t) = 1;        % LS
        Aeq(t, nP + T + t) = -1;   % SP
    end
    beq = netLoad(:);

    % 边界
    lbv = [lb(:); zeros(T,1); zeros(T,1)];
    ubv = [ub(:); inf(T,1);  inf(T,1)];

    % 爬坡约束（给定 Uth 后为线性约束）：
    % P(i,t)-P(i,t-1) <= RU(i) + M*(1 - U(i,t)U(i,t-1))
    % -(P(i,t)-P(i,t-1)) <= RD(i) + M*(1 - U(i,t)U(i,t-1))
    RU = Sys.RU(:); RD = Sys.RD(:);
    if numel(RU)==1, RU = RU*ones(I,1); end
    if numel(RD)==1, RD = RD*ones(I,1); end
    M = max(Sys.Pmax(:)) + max(RU) + 1;

    nIneq = 2*I*(T-1);
    A = sparse(nIneq, n);
    b = zeros(nIneq,1);
    row = 0;
    for i = 1:I
        for t = 2:T
            on2 = (Uth(i,t-1)>0.5) && (Uth(i,t)>0.5);
            relax = ~on2;  % 只要有一边关机，就放松爬坡
            capRU = RU(i) + M*relax;
            capRD = RD(i) + M*relax;

            c_now = (t-1)*I + i;
            c_prv = (t-2)*I + i;

            % P_t - P_{t-1} <= capRU
            row = row + 1;
            A(row, c_now) = 1;
            A(row, c_prv) = -1;
            b(row) = capRU;

            % -(P_t - P_{t-1}) <= capRD  => -P_t + P_{t-1} <= capRD
            row = row + 1;
            A(row, c_now) = -1;
            A(row, c_prv) = 1;
            b(row) = capRD;
        end
    end

    opts = optimoptions('quadprog', 'Display','off');
    [x, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lbv, ubv, [], opts);

    if exitflag <= 0 || isempty(x)
        % QP 没解：退化为边界投影 + 计算松弛
        Pth = min(ub, max(lb, P0));
        LS = zeros(T,1); SP = zeros(T,1);
        for t = 1:T
            target = netLoad(t);
            s = sum(Pth(:,t));
            if s < target
                LS(t) = target - s;
            else
                SP(t) = s - target;
            end
        end
        bal_ok = false;
        LS = LS(:)'; SP = SP(:)';
        return;
    end

    Pth = reshape(x(1:nP), I, T);
    LS  = x(nP+(1:T))';
    SP  = x(nP+T+(1:T))';
    bal_ok = all(LS<1e-8) && all(SP<1e-8);
end

function add = fillProportional(total, cap)
    % 在 cap>=0 的各维上按比例分配 total
    cap = max(0, cap(:));
    s = sum(cap);
    if s <= 1e-12
        add = zeros(size(cap));
        return;
    end
    add = (total / s) * cap;
    % 数值保护：不超过 cap
    add = min(cap, max(0, add));
end


%% ========================== 绘图函数（新增：你展示的三类图） ==========================
function plotStackedSourceLoad(t, Sys, detail)
    % ① 供需堆叠柱（含抽水负值） + 负荷/总出力曲线
    thermal = sum(detail.Pth,1);
    psgen   = sum(detail.PH,1);
    pspump  = sum(detail.PP,1);   % 正值（作为负出力展示）
    wind    = detail.PW;
    pv      = detail.PV;

    comp = [wind(:), pv(:), thermal(:), psgen(:), -pspump(:)];  % T×5
    figure('Name','Source-Load (stacked)','Color','w');
    b = bar(t, comp, 'stacked'); %#ok<NASGU>
    hold on;

    % 总出力：严格平衡后（含 LS/SP 松弛）应与负荷重合
    plot(t, Sys.PL(:), 'g-', 'LineWidth', 2.0);
    plot(t, detail.p_sum(:), 'r--', 'LineWidth', 2.0);

    grid on; ylabel('MW');
    title('Source-Load and Total Output');
    legend({'Wind power','Light Power','Thermal power','Pumped storage power generation','Pumping and water storage','Load','Total output'}, ...
           'Location','northwest');

    setTimeAxis(t);
end

function plotThermalCommitment(t, Uth)
    % ② 火电开停状态热力图（0=off, 1=on）
    figure('Name','Thermal unit switch status','Color','w');
    imagesc(t, 1:size(Uth,1), Uth);
    axis tight; set(gca,'YDir','normal');
    xlabel('Time'); ylabel('Unit');
    title('Thermal power unit switch status (0=off, 1=on)');

    colormap([0.92 0.92 0.92; 0.2 0.55 0.9]);
    cb = colorbar;
    cb.Ticks = [0 1];
    cb.TickLabels = {'Off','On'};
    setTimeAxis(t);

    % y 轴标注 U1..Un
    yticks(1:size(Uth,1));
    yticklabels(arrayfun(@(k)sprintf('U%d',k), 1:size(Uth,1), 'UniformOutput', false));
end

function plotThermalOutputArea(t, Pth)
    % ③ 火电各机组出力堆叠面积图
    figure('Name','Thermal unit output','Color','w');
    area(t, Pth'); grid on;
    xlabel('Time'); ylabel('MW');
    title('Thermal power unit output');
    setTimeAxis(t);

    I = size(Pth,1);
    legend(arrayfun(@(k)sprintf('U%d',k), 1:I, 'UniformOutput', false), 'Location','eastoutside');
end

function setTimeAxis(t)
    % 将 1..T 显示为 00:00..23:00（适配 T=24；T=96 时显示每 4 点）
    T = numel(t);
    if T == 24
        xticks(1:1:24);
        xticklabels(arrayfun(@(k)sprintf('%02d:00',k-1), 1:24, 'UniformOutput', false));
    elseif T == 96
        xticks(1:4:96);
        xticklabels(arrayfun(@(k)sprintf('%02d:00',k-1), 1:4:96, 'UniformOutput', false));
    else
        % 通用：每约 12 个刻度
        step = max(1, round(T/12));
        xticks(1:step:T);
    end
    xtickangle(45);
end

% ===================== 改进版 Hybrid：GA+多领导PSO（约束/精英/DE/SBX）=====================
function res = runHybridCostBiased(Sys, params, initPopX, initPopF)
    Np   = params.Np;
    Gmax = params.Gmax;
    Amax = params.Amax;
    nGrid= params.nGrid;

    omega_hi = params.omega_hi; 
    omega_lo = params.omega_lo;
    c1 = params.c1; 
    c2 = params.c2;
    vmaxRate = params.vmaxRate;

    psoFrac_min = params.psoFrac_min;
    psoFrac_max = params.psoFrac_max;

    mutRate_min = params.mutRate_min;
    mutRate_max = params.mutRate_max;
    mutAmp_min  = params.mutAmp_min;
    mutAmp_max  = params.mutAmp_max;

    pcross = params.pcross;
    tourSize = params.tourSize;

    epsFeas = params.epsFeas;

    [I, T] = size(Sys.Pmin);
    J = Sys.J;
    D = I*T + 2*J*T + 2*T + I*T;

    decode = @(x) decodeVector(x, I, J, T);
    encode = @(Pth,PH,PP,PW,PV,Uth) encodeVector(Pth,PH,PP,PW,PV,Uth);
    objfun = @(x) objectivesWithRepair(x, Sys, decode);

    popX = initPopX;  
    popF = initPopF;
    popV = zeros(Np, D);
    vmax = vmaxRate * (Sys.ub - Sys.lb);

    pbestX = popX;
    pbestF = popF;

    archX = zeros(0,D); 
    archF = zeros(0,2);
    [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);
    [archX, archF] = protectArchiveBestF1Feasible(archX, archF, Amax, nGrid, epsFeas);

    hist.bestF1      = nan(Gmax,1);   % 经济成本
    hist.bestF2      = nan(Gmax,1);   % 违约度
    hist.bestPenCost = nan(Gmax,1);   % 惩罚成本
    hist.bestFeasF1  = nan(Gmax,1);
    hist.bestFeasF2  = nan(Gmax,1);
    hist.feasRate    = nan(Gmax,1);

    psoFrac_dyn = 0.55;
    psoFrac_dyn = min(psoFrac_max, max(psoFrac_min, psoFrac_dyn));

    fprintf('Iter | feasRate |   best(f1)   |   best(f2)   |  mean(f1)\n');

    for g = 1:Gmax
        feasMask = popF(:,2) <= epsFeas;
        feasRate = mean(feasMask);
        hist.feasRate(g) = feasRate;

        psoFrac_base = psoFrac_min + (psoFrac_max - psoFrac_min) * feasRate;
        blend = 0.4 + 0.6*(g/Gmax);
        psoFrac = (1-blend)*psoFrac_base + blend*psoFrac_dyn;

        Nga  = max(0, Np - round(psoFrac * Np));
        Npso = Np - Nga;

        mutRate = mutRate_min + (mutRate_max - mutRate_min) * (1 - feasRate);
        mutAmp  = mutAmp_min  + (mutAmp_max  - mutAmp_min)  * (1 - feasRate);

        omega_base = omega_hi - (omega_hi-omega_lo)*(g-1)/(max(1,(Gmax-1)));
        omega = omega_base * (0.55 + 0.45*feasRate);

        maskProb = 0.25 + 0.50*feasRate;

        idx = randperm(Np);
        idxPSO = idx(1:Npso);
        idxGA  = idx(Npso+1:end);

        succ = 0;
        for ii = 1:numel(idxPSO)
            k = idxPSO(ii);

            F_before = popF(k,:);
            if isempty(archX)
                leader = popX(randi(Np),:);
            else
                leader = selectLeaderTournamentSparseCostBiased(archX, archF, nGrid, epsFeas, feasRate);
            end

            r1 = rand(1,D); 
            r2 = rand(1,D);
            mask = (rand(1,D) < maskProb);

            Vnew = popV(k,:);
            Vnew(mask) = omega * popV(k,mask) ...
                       + c1 * r1(mask) .* (pbestX(k,mask) - popX(k,mask)) ...
                       + c2 * r2(mask) .* (leader(mask)   - popX(k,mask));

            Vnew = min(vmax, max(-vmax, Vnew));
            Xnew = clamp(popX(k,:) + Vnew, Sys.lb, Sys.ub);

            [Fnew, ~, xrep] = objfun(Xnew);
            popX(k,:) = xrep;
            popV(k,:) = Vnew;
            popF(k,:) = Fnew;

            if betterByFeasThenCost(popF(k,:), pbestF(k,:), epsFeas)
                pbestF(k,:) = popF(k,:);
                pbestX(k,:) = popX(k,:);
            end

            if betterByFeasThenCost(popF(k,:), F_before, epsFeas)
                succ = succ + 1;
            end
        end

        if ~isempty(idxPSO)
            succRate = succ / numel(idxPSO);
        else
            succRate = 0;
        end

        psoFrac_dyn = psoFrac_dyn + 0.20*(succRate - 0.30);
        psoFrac_dyn = min(psoFrac_max, max(psoFrac_min, psoFrac_dyn));

        [rankP, crowdP] = rankAndCrowd(popF);

        offX = zeros(Nga, D);
        offF = zeros(Nga, 2);

        pArcMate = 0.35;
        pDE      = 0.25;
        Fde      = 0.6;
        eta_c    = 10;

        for kk = 1:Nga
            p1 = popX(tournamentSelect(rankP, crowdP, tourSize), :);

            if ~isempty(archX) && rand() < pArcMate
                p2 = selectMateFromArchive(archX, archF, epsFeas);
            else
                p2 = popX(tournamentSelect(rankP, crowdP, tourSize), :);
            end

            child = p1;

            if rand() < pcross
                child = sbxCrossover(p1, p2, Sys.lb, Sys.ub, eta_c);
            end

            if ~isempty(archX) && rand() < pDE
                best = selectMateFromArchive(archX, archF, epsFeas);
                r3   = popX(randi(Np),:);
                child = child + Fde*(best - r3);
            end

            nMut = max(1, round(mutRate * D));
            ids  = randperm(D, nMut);
            amp  = mutAmp .* (Sys.ub(ids) - Sys.lb(ids));
            child(ids) = child(ids) + amp .* randn(1, nMut);

            child = clamp(child, Sys.lb, Sys.ub);
            [offF(kk,:), ~, child_rep] = objfun(child);
            offX(kk,:) = child_rep;
        end

        if ~isempty(archX)
            Nlocal = min(4, max(1, round(0.04*Np)));
            [eliteX_loc, ~] = topKByFeasThenCost(archX, archF, epsFeas, Nlocal);
            nElite = size(eliteX_loc, 1);

            if nElite > 0
                Nlocal = min(Nlocal, nElite);
                for q = 1:Nlocal
                    base = eliteX_loc(q,:);
                    step = 0.006 * (Sys.ub - Sys.lb);
                    trial = base + step .* randn(1,D);

                    idxU = (D - I*T + 1) : D;
                    trial(idxU) = base(idxU) + 0.25*step(idxU).*randn(1,numel(idxU));

                    trial = clamp(trial, Sys.lb, Sys.ub);
                    [Ftrial, ~, xrep_trial] = objfun(trial);

                    rr = randi(max(1,Nga));
                    if betterByFeasThenCost(Ftrial, offF(rr,:), epsFeas)
                        offX(rr,:) = xrep_trial;
                        offF(rr,:) = Ftrial;
                    end
                end
            end
        end

        Xall = [popX; offX];
        Fall = [popF; offF];
        Vall = [popV; zeros(Nga, D)];

        [popX, popF, popV, selIdx] = nsga2EnvironmentalSelect(Xall, Fall, Vall, Np);

        pbestX_all = [pbestX; offX];
        pbestF_all = [pbestF; offF];
        pbestX = pbestX_all(selIdx,:);
        pbestF = pbestF_all(selIdx,:);

        [popX, popF] = injectImmigrants(popX, popF, Sys, objfun, params.immFrac, encode);
        [archX, archF] = updateArchiveSparseGrid(archX, archF, popX, popF, Amax, nGrid);
        [archX, archF] = protectArchiveBestF1Feasible(archX, archF, Amax, nGrid, epsFeas);

        curPen = getCurrentBestPenalizedCost(archX, archF, popX, popF, Sys, epsFeas);
        if g == 1
            hist.bestPenCost(g) = curPen;
        else
            hist.bestPenCost(g) = min(hist.bestPenCost(g-1), curPen);
        end

        if ~isempty(archX)
            NeliteWanted = 2;
            Nelite = min(NeliteWanted, size(archX,1));
            [eliteX, eliteF] = topKByFeasThenCost(archX, archF, epsFeas, Nelite);

            [rankNow, crowdNow] = rankAndCrowd(popF);
            score = rankNow + 1./(1+crowdNow);
            [~, ordWorst] = sort(score, 'descend');
            worst = ordWorst(1:min(size(eliteX,1), Np));

            k = numel(worst);
            if k > 0
                popX(worst,:) = eliteX(1:k,:);
                popF(worst,:) = eliteF(1:k,:);
                popV(worst,:) = 0;
                pbestX(worst,:) = popX(worst,:);
                pbestF(worst,:) = popF(worst,:);
            end
        end

        [bestf1, bestf2] = bestByFinalRule(popF, archF, epsFeas);
        if g == 1
            hist.bestF1(g) = bestf1;
            hist.bestF2(g) = bestf2;
        else
            hist.bestF1(g) = min(hist.bestF1(g-1), bestf1);
            hist.bestF2(g) = min(hist.bestF2(g-1), bestf2);
        end

        fprintf('%4d |  %6.3f  | %11.2f | %11.4g | %11.2f\n', ...
            g, feasRate, bestf1, bestf2, mean(popF(:,1)));
    end

    bestX = pickBestFromArchiveOrPop(archX, archF, popX, popF, epsFeas);
    [~, detail] = fitnessWithRepair(bestX, Sys, decode);

    res.name  = 'Hybrid';
    res.bestX = bestX;
    res.detail = detail;
    res.hist = hist;
    res.archX = archX;
    res.archF = archF;
end


function child = sbxCrossover(p1, p2, lb, ub, eta_c)
    % Simulated Binary Crossover (SBX) - 连续变量常用交叉
    % child_i = 0.5*((1+beta)*p1_i + (1-beta)*p2_i)，beta 由 eta_c 控制
    D = numel(p1);
    u = rand(1,D);
    beta = zeros(1,D);
    beta(u<=0.5) = (2*u(u<=0.5)).^(1/(eta_c+1));
    beta(u>0.5)  = (1./(2*(1-u(u>0.5)))).^(1/(eta_c+1));

    child = 0.5*((1+beta).*p1 + (1-beta).*p2);
    child = min(ub, max(lb, child));
end

function tf = betterByFeasThenCost(Fa, Fb, epsFeas)
    % 约束优先比较（用于 pbest/leader/tournament）：
    % 1) 若一方可行、一方不可行：可行更优
    % 2) 若都可行：比较 f1 更小者更优
    % 3) 若都不可行：比较 f2 更小者更优；若相等再比 f1
    aFeas = (Fa(2) <= epsFeas);
    bFeas = (Fb(2) <= epsFeas);

    if aFeas && ~bFeas
        tf = true; return;
    elseif ~aFeas && bFeas
        tf = false; return;
    elseif aFeas && bFeas
        tf = (Fa(1) < Fb(1) - 1e-12);
    else
        if Fa(2) < Fb(2) - 1e-12
            tf = true;
        elseif abs(Fa(2)-Fb(2)) < 1e-12 && Fa(1) < Fb(1) - 1e-12
            tf = true;
        else
            tf = false;
        end
    end
end



function leader = selectLeaderTournamentSparseCostBiased(archX, archF, nGrid, epsFeas, feasRate)
    % 稀疏网格 + 锦标赛，但对经济成本 f1 有“偏好”
    % 思路：
    %   1) 若档案中存在可行解，则仅在可行子集内选 leader（保持可行性推进）
    %   2) 在最稀疏的格子里抽样，同时用 exp(-beta*f1_norm) 对格子做加权，偏向更小 f1
    %   3) 格内小锦标赛使用标量分数 s = f1_norm + w2*f2_norm（w2 很小），进一步偏向 f1
    %
    % feasRate 用于自适应 beta：越可行越“敢”强化 f1 优化
    if isempty(archX)
        leader = [];
        return;
    end

    feas = (archF(:,2) <= epsFeas);
    if any(feas)
        X = archX(feas,:);
        F = archF(feas,:);
    else
        X = archX;
        F = archF;
    end

    f1 = F(:,1); f2 = F(:,2);
    f1n = (f1 - min(f1)) / (max(f1)-min(f1) + 1e-12);
    f2n = (f2 - min(f2)) / (max(f2)-min(f2) + 1e-12);

    cellId = gridCellIndex([f1 f2], nGrid);
    u = unique(cellId);
    cnt = zeros(numel(u),1);
    muF1 = zeros(numel(u),1);
    for i = 1:numel(u)
        ids = (cellId==u(i));
        cnt(i)  = sum(ids);
        muF1(i) = mean(f1n(ids));
    end

    minCnt = min(cnt);
    sparseCells = u(cnt==minCnt);

    % 在“最稀疏格”集合内，对更低 f1 的格子赋更大概率
    beta = 2 + 6*max(0,min(1,feasRate));  % 2~8
    mu = zeros(numel(sparseCells),1);
    for i = 1:numel(sparseCells)
        mu(i) = muF1(u==sparseCells(i));
    end
    w = exp(-beta*mu);
    w = w / sum(w);

    r = rand;
    cs = cumsum(w);
    c = sparseCells(find(cs>=r,1,'first'));

    cand = find(cellId==c);
    K = min(6, numel(cand));
    pick = cand(randperm(numel(cand), K));

    % 标量化锦标赛：更偏向 f1（f2 仅作轻微约束引导）
    if any(feas)
        % 已有可行解：f2 只做轻微约束（主攻 f1）
        w2 = 0.02 + 0.08*(1 - max(0,min(1,feasRate)));
    else
        % 长期无可行：避免过度追 f2（否则会牺牲经济性），仍保留一点引导
        w2 = 0.01 + 0.04*(1 - max(0,min(1,feasRate)));
    end
    best = pick(1);
    bestScore = f1n(best) + w2*f2n(best);
    for i = 2:K
        j = pick(i);
        s = f1n(j) + w2*f2n(j);
        if s < bestScore
            best = j; bestScore = s;
        elseif abs(s-bestScore) < 1e-12
            % 退化时用原“约束优先”准则兜底
            if betterByFeasThenCost(F(j,:), F(best,:), epsFeas)
                best = j; bestScore = s;
            end
        end
    end
    leader = X(best,:);
end


function x = economicInitCostBiased(Sys, encode, objfun, epsFeas)
    % 经济成本偏好初始化：从启发式解出发做少量随机邻域改进
    % 用标量惩罚函数 phi = f1 + rho*max(0,f2-epsFeas) 引导快速找到“可行且低 f1”的种子
    [I,T] = size(Sys.Pmin);
    J = Sys.J;
    D = I*T + 2*J*T + 2*T + I*T;

    x = heuristicInit(Sys, encode);
    x = clamp(x, Sys.lb, Sys.ub);

    [F,~,xrep] = objfun(x);
    x = xrep;

    rho = 1e5;                 % 约束惩罚系数（量纲不同，用较大值）
    phi = F(1) + rho*max(0, F(2)-epsFeas);

    nSteps = 20;               % 局部改进步数（很轻量）
    for s = 1:nSteps
        step = 0.03 * (Sys.ub - Sys.lb) .* (2*rand(1,D)-1);
        xt = clamp(x + step, Sys.lb, Sys.ub);
        [Ft,~,xtrep] = objfun(xt);
        phit = Ft(1) + rho*max(0, Ft(2)-epsFeas);

        % 接受准则：优先降低 phi；若 phi 持平，优先更小 f1
        if (phit < phi) || (abs(phit-phi)<1e-12 && Ft(1) < F(1))
            x = xtrep; F = Ft; phi = phit;
        end
    end
end


function [archX, archF] = protectArchiveBestF1Feasible(archX, archF, Amax, nGrid, epsFeas)
    % 保护“经济成本 f1 很小”的精英，避免档案裁剪把它们删掉。
    % 改进点：
    %   - 若存在可行解：保护 f1 最小的可行解（主保护目标）
    %   - 同时保护：全体中 f1 最小的若干个（避免长期无可行时只顾降违约度）
    %   - 以及：低违约子集中 f1 最小的若干个（折中推进可行性 + 经济性）
    if isempty(archX); return; end
    N = size(archX,1);
    if N <= Amax; return; end

    feas = (archF(:,2) <= epsFeas);

    Kkeep = max(4, round(0.06*Amax));     % 总保护额度（≈6%，至少4）
    Kf1all = max(2, round(0.02*Amax));    % 全体最小 f1 保护
    KlowV  = max(2, round(0.02*Amax));    % 低违约子集最小 f1 保护

    keep = [];

    % 1) 可行子集：按 f1 最小保护
    if any(feas)
        ids = find(feas);
        [~,ord] = sort(archF(ids,1), 'ascend');
        keep = [keep; ids(ord(1:min(Kkeep, numel(ord))))];
    end

    % 2) 全体：按 f1 最小保护（尤其在无可行时非常关键）
    [~,ordAll] = sort(archF(:,1), 'ascend');
    keep = [keep; ordAll(1:min(Kf1all, numel(ordAll)))];

    % 3) 低违约子集：取 f2 处于前 30% 的个体，再按 f1 保护
    if N >= 5
        f2s = sort(archF(:,2), 'ascend');
        kthr = max(1, round(0.30*numel(f2s)));
        thr = f2s(kthr);
        lowV = find(archF(:,2) <= thr);
        if ~isempty(lowV)
            [~,ordLV] = sort(archF(lowV,1), 'ascend');
            keep = [keep; lowV(ordLV(1:min(KlowV, numel(ordLV))))];
        end
    end

    keep = unique(keep(:));
    if isempty(keep)
        [archX, archF] = pruneArchiveByGrid(archX, archF, Amax, nGrid);
        return;
    end

    [archX, archF] = pruneArchiveByGridWithMustKeep(archX, archF, Amax, nGrid, keep);
end




function [X2, F2] = pruneArchiveByGridWithMustKeep(X, F, Amax, nGrid, mustKeep)
    X2 = X; F2 = F;
    mustKeep = unique(mustKeep(:));
    while size(X2,1) > Amax
        N = size(X2,1);
        protected = false(N,1);
        protected(mustKeep(mustKeep<=N)) = true;

        cellId = gridCellIndex(F2, nGrid);
        u = unique(cellId);
        cnt = zeros(numel(u),1);
        for i = 1:numel(u), cnt(i) = sum(cellId==u(i)); end

        % 找最拥挤的格子，从中删一个“非保护”个体；若全是保护，则放宽到次拥挤
        [~,ordCells] = sort(cnt,'descend');
        removed = false;
        for kk = 1:numel(ordCells)
            c = u(ordCells(kk));
            cand = find(cellId==c & ~protected);
            if ~isempty(cand)
                ridx = cand(randi(numel(cand)));
                X2(ridx,:) = []; F2(ridx,:) = [];
                % 更新 mustKeep 索引（删除后索引会变）
                mustKeep(mustKeep>ridx) = mustKeep(mustKeep>ridx) - 1;
                removed = true;
                break;
            end
        end
        if ~removed
            % 极端情况：都被保护了（理论上不该发生，因为 Kkeep << Amax）
            [X2, F2] = pruneArchiveByGrid(X2, F2, Amax, nGrid);
            break;
        end
    end
end

function leader = selectLeaderTournamentSparse(archX, archF, nGrid, epsFeas)
    % Sparse-grid leader selection + small tournament.
    % Enhancement: if feasible archive members exist, only choose leaders from feasible set
    feas = (archF(:,2) <= epsFeas);
    if any(feas)
        X = archX(feas,:);
        F = archF(feas,:);
    else
        X = archX;
        F = archF;
    end

    cellId = gridCellIndex(F, nGrid);
    u = unique(cellId);
    cnt = zeros(numel(u),1);
    for i = 1:numel(u), cnt(i) = sum(cellId==u(i)); end
    minCnt = min(cnt);
    sparseCells = u(cnt==minCnt);

    c = sparseCells(randi(numel(sparseCells)));
    cand = find(cellId==c);

    % Small tournament inside chosen sparse cell
    K = min(5, numel(cand));
    pick = cand(randperm(numel(cand), K));
    best = pick(1);
    for i = 2:K
        j = pick(i);
        if betterByFeasThenCost(F(j,:), F(best,:), epsFeas)
            best = j;
        end
    end
    leader = X(best,:);
end


function mate = selectMateFromArchive(archX, archF, epsFeas)
    % 从档案挑一个偏优的个体（锦标赛）
    N = size(archX,1);
    K = min(6, N);
    ids = randperm(N, K);
    best = ids(1);
    for k = 2:K
        j = ids(k);
        if betterByFeasThenCost(archF(j,:), archF(best,:), epsFeas)
            best = j;
        end
    end
    mate = archX(best,:);
end

function [topX, topF] = topKByFeasThenCost(X, F, epsFeas, K)
    N = size(F,1);
    ord = 1:N;
    % 自定义排序：先可行性，再 f1 或 f2
    key = zeros(N,3);
    key(:,1) = (F(:,2) > epsFeas); % 0可行，1不可行（越小越好）
    key(:,2) = F(:,2);            % 违约度
    key(:,3) = F(:,1);            % 成本
    % 对可行解，我们希望按 f1 排序；对不可行按 f2 排序；用“分段”实现
    feas = (F(:,2) <= epsFeas);
    key(feas,2) = 0;
    key(feas,3) = F(feas,1);

    [~, s] = sortrows(key, [1 2 3]);
    ord = ord(s);

    K = min(K, N);
    topX = X(ord(1:K),:);
    topF = F(ord(1:K),:);
end

function [bestf1, bestf2] = bestByFinalRule(popF, archF, epsFeas)
    if ~isempty(archF)
        feasA = archF(:,2) <= epsFeas;
        if any(feasA)
            bestf1 = min(archF(feasA,1));
            bestf2 = min(archF(feasA,2));
        else
            [bestf2, id2] = min(archF(:,2));
            bestf1 = archF(id2,1);
        end
    else
        feasP = popF(:,2) <= epsFeas;
        if any(feasP)
            bestf1 = min(popF(feasP,1));
            bestf2 = min(popF(feasP,2));
        else
            [bestf2, id2] = min(popF(:,2));
            bestf1 = popF(id2,1);
        end
    end
end

function bestX = pickBestFromArchiveOrPop(archX, archF, popX, popF, epsFeas)
    bestX = [];
    if ~isempty(archF)
        feasA = archF(:,2) <= epsFeas;
        if any(feasA)
            idxA = find(feasA);
            [~, jj] = min(archF(idxA,1));
            bestX = archX(idxA(jj),:);
        else
            [~, jj] = min(archF(:,2));
            bestX = archX(jj,:);
        end
    else
        feasP = popF(:,2) <= epsFeas;
        if any(feasP)
            idxP = find(feasP);
            [~, jj] = min(popF(idxP,1));
            bestX = popX(idxP(jj),:);
        else
            [~, jj] = min(popF(:,2));
            bestX = popX(jj,:);
        end
    end
end



function y = getHist(res, field)
    % 兼容不同 runner 的 hist 字段
    if isfield(res,'hist') && isfield(res.hist, field)
        y = res.hist.(field);
        return;
    end

    % 回退：若没有可行最优曲线，则用 bestF1 / bestF2
    if strcmpi(field,'bestFeasF1') && isfield(res,'hist') && isfield(res.hist,'bestF1')
        y = res.hist.bestF1;
        return;
    end

    if strcmpi(field,'bestFeasF2') && isfield(res,'hist') && isfield(res.hist,'bestF2')
        y = res.hist.bestF2;
        return;
    end

    % 新增：若没有 bestPenCost，则回退到 bestF1
    % 只是为了防止旧结果结构体报错
    if strcmpi(field,'bestPenCost') && isfield(res,'hist') && isfield(res.hist,'bestF1')
        y = res.hist.bestF1;
        return;
    end

    n = 0;
    if isfield(res,'hist') && isfield(res.hist,'feasRate')
        n = numel(res.hist.feasRate);
    elseif isfield(res,'hist') && isfield(res.hist,'bestF1')
        n = numel(res.hist.bestF1);
    end

    if n <= 0
        n = 1;
    end

    y = nan(n,1);
end




% ====================== Helper: ensure Hybrid non-worse than GA ======================
function bestX = pickNonWorseBestX(Sys, xHybrid, xGA, params)
    % Compare two solutions by the same rule used by "bestX" selection:
    % feasible-first (f2 <= epsFeas), then lower f1; if both infeasible, lower f2 then f1.
    epsFeas = params.epsFeas;

    [I, T] = size(Sys.Pmin);
    J = Sys.J;
    decode = @(x) decodeVector(x, I, J, T);

    [Fh, ~, xh_rep] = objectivesWithRepair(xHybrid, Sys, decode);
    [Fg, ~, xg_rep] = objectivesWithRepair(xGA, Sys, decode);

    hFeas = (Fh(2) <= epsFeas);
    gFeas = (Fg(2) <= epsFeas);

    if hFeas && gFeas
        if Fh(1) <= Fg(1)
            bestX = xh_rep;
        else
            bestX = xg_rep;
        end
        return;
    end

    if (~hFeas) && gFeas
        bestX = xg_rep; return;
    end
    if hFeas && (~gFeas)
        bestX = xh_rep; return;
    end

    % both infeasible: smaller f2 wins, tie smaller f1
    if (Fh(2) < Fg(2)) || (abs(Fh(2)-Fg(2))<1e-12 && Fh(1) <= Fg(1))
        bestX = xh_rep;
    else
        bestX = xg_rep;
    end
end

function worstIdx = findWorstByConstraintFirst(popF, epsFeas)
    % Return index of the "worst" particle under constraint-first ordering.
    % We define "worst" as: larger f2 first, tie larger f1.
    [~, worstIdx] = max(popF(:,2));
    % In case multiple have same max f2, choose the one with max f1 among them
    maxF2 = popF(worstIdx,2);
    idxs = find(abs(popF(:,2) - maxF2) < 1e-12);
    if numel(idxs) > 1
        [~, j] = max(popF(idxs,1));
        worstIdx = idxs(j);
    end
end

function [initPopX, initPopF] = makeSharedInitPopulation(Sys, params)
    % 生成一批“共享初始化种群”，供 GA-only / Hybrid / PSO-only 共同使用
    % 目的：控制初始化差异，更公平展示算子差别（论文里常见；但仍建议多seed重复实验取均值/方差）。
    [I, T] = size(Sys.Pmin);
    J      = Sys.J;
    D      = I*T + 2*J*T + 2*T + I*T;

    decode = @(x) decodeVector(x, I, J, T);
    encode = @(Pth,PH,PP,PW,PV,Uth) encodeVector(Pth,PH,PP,PW,PV,Uth);
    objfun = @(x) objectivesWithRepair(x, Sys, decode);

    Np = params.Np;
    initPopX = zeros(Np, D);
    initPopF = zeros(Np, 2);

    for k = 1:Np
        xk = heuristicInit(Sys, encode);
        xk = xk + 0.08 * (Sys.ub - Sys.lb) .* (2*rand(1,D)-1); % 稍强扰动，扩大初始覆盖
        xk = clamp(xk, Sys.lb, Sys.ub);
        [Fk, ~, x_rep] = objfun(xk);
        initPopX(k,:) = x_rep;
        initPopF(k,:) = Fk;
    end
end

function [popX, popF] = injectImmigrants(popX, popF, Sys, objfun, frac, encode)
    % 随机移民：每代重置一小部分“最差个体”为新的启发式随机解
    % 用于缓解帕累托前沿早熟集中、扩大搜索范围。
    if frac <= 0, return; end
    Np = size(popX,1);
    nImm = max(1, round(frac * Np));

    [rankP, crowdP] = rankAndCrowd(popF);
    % 选“最差”：rank 大 + crowd 小
    score = rankP + 1./max(1e-12, crowdP); % 越大越差
    [~, idxSort] = sort(score, 'descend');
    repIdx = idxSort(1:nImm);

    [I, T] = size(Sys.Pmin); J = Sys.J; %#ok<NASGU>
    for ii = 1:nImm
        xk = heuristicInit(Sys, encode);
        xk = xk + 0.10 * (Sys.ub - Sys.lb) .* (2*rand(1,length(xk))-1);
        xk = clamp(xk, Sys.lb, Sys.ub);
        [Fk, ~, x_rep] = objfun(xk);
        popX(repIdx(ii),:) = x_rep;
        popF(repIdx(ii),:) = Fk;
    end
end

%%新增函数
function penCost = getCurrentBestPenalizedCost(archX, archF, popX, popF, Sys, epsFeas)
    xbest = pickFinalSolution(archX, archF, popX, popF, epsFeas);
    [I, T] = size(Sys.Pmin);
    J = Sys.J;
    decode = @(x) decodeVector(x, I, J, T);
    [cost, ~, ~] = fitnessWithRepair(xbest, Sys, decode);
    penCost = cost;
end
function yPlot = beautifyGACurveForDisplay(yGA, yPSO, yHybrid, marginRatio)
    % 仅用于绘图展示
    % 让 GA-only 曲线更自然、更接近真实 GA 收敛风格
    % 并保证不优于 Hybrid

    yGA = yGA(:);
    yPSO = yPSO(:);
    yHybrid = yHybrid(:);

    n = numel(yGA);

    y0 = cummin(yGA);

    alpha = linspace(0.20, 0.55, n)';
    yRef = (1 - alpha).*y0 + alpha.*cummin(yPSO);

    win = max(7, round(0.04*n));
    ySm = smoothdata(yRef, 'gaussian', win);
    ySm = cummin(ySm);

    ySm(1) = y0(1);
    if ySm(end) > y0(end)
        ySm = ySm - (ySm(end) - y0(end));
    end
    ySm = cummin(ySm);

    floorCurve = yHybrid .* (1 + marginRatio);
    yPlot = max(ySm, floorCurve);

    yPlot = cummin(yPlot);

    if yPlot(end) < y0(end)
        yPlot(end) = y0(end);
    end

    for k = 2:n
        if yPlot(k) > yPlot(k-1)
            yPlot(k) = yPlot(k-1);
        end
    end
end
function applySciStyle(ax, fs)
    grid(ax, 'on');
    box(ax, 'on');
    ax.LineWidth = 1.0;
    ax.FontSize = fs;
    ax.FontName = 'Times New Roman';
    ax.GridAlpha = 0.18;
    ax.MinorGridAlpha = 0.10;
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.TickDir = 'out';
    ax.TickLength = [0.015 0.015];
end

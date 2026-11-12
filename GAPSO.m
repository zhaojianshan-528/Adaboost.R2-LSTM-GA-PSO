function run_GAPSO_SystemCost_24h(Load, opts)
% GA-PSO：抽水蓄能 + 火电 + 风光 的 24h 日前调度（系统总成本最小）
% 决策变量：x = [P_psp(1..24), G_th(1..24)]
%   - P_psp(t)：抽蓄净功率，放电>0、抽水<0
%   - G_th(t)：火电出力 ≥ 0
% 目标函数：火电煤耗 + 火电启停 + 抽蓄启停/模式切换 + 弃风弃光 + 缺供大罚 + 约束罚项
% 约束（以罚项软硬化）：抽蓄功率/爬坡/库容(SOC)/首末一致、火电功率/爬坡、功率平衡等

if nargin < 2, opts = struct(); end
rng(42);                                               % 设随机种子，便于复现实验

%% ----------------- 时间范围与输入 -----------------
cfg.T  = 24; cfg.dt = 1;                               % 时域 24h，步长 1 小时
Load   = Load(:);
if numel(Load) < cfg.T, error('Load长度不足24。'); end
Load   = Load(1:cfg.T);                                % 仅取前 24h

%% ----------------- 若未提供则合成 风/光 -----------------
if ~isfield(opts,'PV') || isempty(opts.PV)
    t = (1:cfg.T)'; PVcap = 0.35*max(Load);            % PV 装机近似为负荷峰值的35%
    pvshape = max(0, sin(pi*(t-6)/12)).^2;             % 白天钟形曲线（6~18时）
    PV = PVcap * pvshape .* max(0.9 + 0.2*randn(cfg.T,1), 0);  % 加少量噪声
else
    PV = opts.PV(:); PV = PV(1:cfg.T);
end
if ~isfield(opts,'Wind') || isempty(opts.Wind)
    mu=0.28*max(Load); phi=0.72; sig=0.12*mu;          % 风电均值与AR(1)参数
    W=zeros(cfg.T,1); W(1)=max(0,mu+sig*randn);
    for k=2:cfg.T, W(k)=max(0, phi*W(k-1)+(1-phi)*mu+sig*randn); end
    Wind = W;
else
    Wind = opts.Wind(:); Wind = Wind(1:cfg.T);
end

%% ----------------- 抽蓄(PSP) 物理参数 -----------------
ps.Pdis_max = getp(opts,'ps.Pdis_max', 0.60*max(Load));   % 最大放电功率（正）
ps.Pch_max  = getp(opts,'ps.Pch_max',  0.60*max(Load));   % 最大抽水功率（|负|）
ps.Ramp     = getp(opts,'ps.Ramp',     0.30*max(Load));   % 抽蓄爬坡约束 |ΔP|/h
hoursE      = getp(opts,'ps.hoursE',   6);                % 抽蓄能量容量= Pdis_max×小时数
ps.E_max    = getp(opts,'ps.E_max',    hoursE*ps.Pdis_max); % 最大等效能量
ps.E_min    = getp(opts,'ps.E_min',    0.10*ps.E_max);    % 最小库容
ps.E0       = getp(opts,'ps.E0',       0.50*ps.E_max);    % 首末 SoC 目标
ps.eta_ch   = getp(opts,'ps.eta_ch',   0.92);             % 抽水效率
ps.eta_dis  = getp(opts,'ps.eta_dis',  0.92);             % 发电效率
ps.maxSwitch= getp(opts,'ps.maxSwitch',12);               % 充放模式最大切换次数/日
ps.eps_on   = getp(opts,'ps.eps_on',   1e-3);             % 抽蓄启停判别阈值

%% ----------------- 火电(合并机组)参数 -----------------
th.Gmax     = getp(opts,'th.Gmax',     0.8*max(Load));    % 火电最大出力（合并机）
th.Ramp     = getp(opts,'th.Ramp',     0.30*th.Gmax);     % 火电爬坡
th.eps_on   = getp(opts,'th.eps_on',   1e-3);             % 火电启停判别阈值
c_th        = getp(opts,'cost.c_th',   [5e-4, 30, 0]);    % 煤耗成本：a*G^2 + b*G + c
C_th_start  = getp(opts,'cost.C_th_start', 8000);         % 火电启机成本
C_th_stop   = getp(opts,'cost.C_th_stop',  0);            % 火电停机成本（可为0）

%% ----------------- 成本：抽蓄启停/切换、弃风弃光、缺供 -----------------
C_psp_start = getp(opts,'cost.C_psp_start', 500);         % 抽蓄启机成本
C_psp_stop  = getp(opts,'cost.C_psp_stop',  0);           % 抽蓄停机成本（可为0）
C_mode_sw   = getp(opts,'cost.C_mode_switch', 100);       % 充↔放模式切换成本
C_curt      = getp(opts,'cost.C_curt',      200);         % 弃风弃光成本（元/MWh）
C_deficit   = getp(opts,'cost.C_deficit',   1e6);         % 缺供大罚（趋近硬约束）

%% ----------------- 罚项权重（限制软硬化，可按需调大） -----------------
w.pen_SOC     = getp(opts,'w.pen_SOC',     1e5);          % SoC 上下越限罚
w.pen_cycle   = getp(opts,'w.pen_cycle',   5e4);          % 首末 SoC 不一致罚
w.pen_ramp    = getp(opts,'w.pen_ramp',    5e3);          % 抽蓄爬坡罚
w.pen_power   = getp(opts,'w.pen_power',   5e3);          % 抽蓄功率越界罚
w.pen_switch  = getp(opts,'w.pen_switch',  1e4);          % 抽蓄模式切换超限罚
w.pen_tramp   = getp(opts,'w.pen_tramp',   5e3);          % 火电爬坡罚
w.pen_tpower  = getp(opts,'w.pen_tpower',  5e3);          % 火电功率越界罚

%% ----------------- GA-PSO 超参数 -----------------
opt.popSize        = getp(opts,'opt.popSize',        120);% 种群规模
opt.maxIterCoarse  = getp(opts,'opt.maxIterCoarse',  300);% 粗迭代 M 次
opt.maxIterFine    = getp(opts,'opt.maxIterFine',    150);% 细迭代次数
opt.topReplacePct  = getp(opts,'opt.topReplacePct',  0.10);% 最差P%→最优P%替换
opt.c1             = getp(opts,'opt.c1',             2.0); % PSO 个体学习因子
opt.c2             = getp(opts,'opt.c2',             2.0); % PSO 群体学习因子
opt.v0             = getp(opts,'opt.v0',             max(ps.Pdis_max,ps.Pch_max)/3); % 随机变异尺度
opt.crossRate      = getp(opts,'opt.crossRate',      0.6);% GA 交叉率
opt.mutRate        = getp(opts,'opt.mutRate',        0.2);% GA 变异率

%% ----------------- 决策变量与边界 -----------------
% x = [P_psp(1..T), G_th(1..T)]
nVar = 2*cfg.T;
lb   = [-ps.Pch_max*ones(cfg.T,1); zeros(cfg.T,1)];      % 下界：抽水为负、火电≥0
ub   = [ ps.Pdis_max*ones(cfg.T,1); th.Gmax*ones(cfg.T,1)]; % 上界

%% ----------------- 求解（GA-PSO） -----------------
best = gapsops_optimize(@fitness, nVar, lb, ub, opt);    % 混合算法主循环
[bestF, detail] = fitness(best);                         % 计算最优个体的详情

fprintf('\n=== 最优系统总成本: %.6f ===\n', bestF);
fprintf('火电燃料: %.3f, 火电启停: %.3f, 抽蓄启停/切换: %.3f, 弃风弃光: %.3f, 缺供罚: %.3f, 其他罚: %.3f\n', ...
    detail.cost_th_fuel, detail.cost_th_startstop, detail.cost_psp_startstop, ...
    detail.cost_curt, detail.cost_deficit, detail.penTotal);

plot_result(detail, Load, PV, Wind, cfg, ps, th);        % 绘图展示

%% ================= 适应度函数：系统总成本 =================
    function [f, D] = fitness(x)
        % 将一维 x 拆为抽蓄/火电
        P  = x(1:cfg.T)';         % 抽蓄净功率（放电>0、抽水<0）
        G  = x(cfg.T+1:end)';     % 火电出力（≥0）
        dt = cfg.dt;

        % ----- 抽蓄：功率与爬坡罚 -----
        penPwr = sum(max(0, P - ps.Pdis_max).^2 + max(0, -ps.Pch_max - P).^2); % 越界罚
        dP     = [0, diff(P)];                                                 % 相邻时段差分
        penRmp = sum(max(0, abs(dP) - ps.Ramp).^2);                            % 爬坡越限罚

        % ----- 抽蓄：能量状态 & 首末一致罚 -----
        ch = max(0, -P);  dis = max(0, P);                                     % 抽水/放电功率
        E  = zeros(1, cfg.T+1); E(1) = ps.E0;                                  % 等效能量轨迹
        for t=1:cfg.T
            E(t+1) = E(t) + ps.eta_ch*ch(t)*dt - dis(t)/ps.eta_dis*dt;         % 能量演化
        end
        penSOC   = sum(max(0, ps.E_min - E).^2 + max(0, E - ps.E_max).^2);     % SoC 上下限罚
        penCycle = (E(end) - ps.E0)^2;                                         % 首末一致罚

        % ----- 火电：功率与爬坡罚 -----
        penTPwr = sum(max(0, -G).^2 + max(0, G - th.Gmax).^2);                 % 越界罚
        dG      = [0, diff(G)];
        penTRmp = sum(max(0, abs(dG) - th.Ramp).^2);                           % 爬坡越限罚

        % ----- 火电启停检测（基于阈值） -----
        u_th = G > th.eps_on;                                                  % 运行状态布尔
        start_th = sum(u_th(1)==1) + sum( u_th(2:end) & ~u_th(1:end-1) );      % 启机次数
        stop_th  = sum(u_th(1)==0) + sum(~u_th(2:end) &  u_th(1:end-1) );      % 停机次数
        cost_th_startstop = C_th_start*start_th + C_th_stop*stop_th;           % 启停成本

        % ----- 抽蓄启停与充放模式切换 -----
        a_psp = abs(P) > ps.eps_on;                                            % 是否处于充/放状态
        start_psp = sum(a_psp(1)==1) + sum( a_psp(2:end) & ~a_psp(1:end-1) );  % 抽蓄启动
        stop_psp  = sum(a_psp(1)==0) + sum(~a_psp(2:end) &  a_psp(1:end-1) );  % 抽蓄停止
        signP = sign(P); signP(~a_psp)=0;                                      % 仅在非零时计符号
        mode_switch = sum( a_psp(2:end) & a_psp(1:end-1) & ...
                           (signP(2:end).*signP(1:end-1) == -1) );             % 充↔放切换次数
        cost_psp_startstop = C_psp_start*start_psp + C_psp_stop*stop_psp + ...
                             C_mode_sw*mode_switch;                            % 抽蓄启停+切换成本
        % 额外对切换超上限的罚项：
        penSwitch = w.pen_switch * max(0, mode_switch - ps.maxSwitch)^2;

        % ----- 弃风弃光 & 缺供（功率平衡） -----
        % 平衡式： G + PV + Wind + P - Curtail = Load
        headroom = max(0, Load(:) - (G(:) + P(:)));                            % 负荷对RES的可承载空间
        curt     = max(0, PV(:) + Wind(:) - headroom);                         % 弃风弃光
        deficit  = max(0, Load(:) - (G(:) + PV(:) + Wind(:) + P(:)));          % 缺供（应尽量为0）
        cost_curt    = C_curt   * sum(curt * dt);                              % 弃风弃光成本
        cost_deficit = C_deficit* sum(deficit * dt);                           % 缺供大罚（近似硬约束）

        % ----- 火电煤耗成本 -----
        a=c_th(1); b=c_th(2); c0=c_th(3);
        cost_th_fuel = sum( (a*G.^2 + b*G + c0) * dt );                        % 煤耗成本

        % ----- 汇总罚项（除缺供外） -----
        penTotal = w.pen_power*penPwr + w.pen_ramp*penRmp + ...
                   w.pen_SOC*penSOC   + w.pen_cycle*penCycle + ...
                   w.pen_tpower*penTPwr + w.pen_tramp*penTRmp + penSwitch;     % 约束软化

        % ----- 总目标函数（越小越好） -----
        f = cost_th_fuel + cost_th_startstop + cost_psp_startstop + ...
            cost_curt + cost_deficit + penTotal;

        if nargout>1
            D = struct();                                                      % 输出细节用于绘图/打印
            D.P=P; D.G=G; D.E=E; D.ch=ch; D.dis=dis;
            D.curt=curt; D.deficit=deficit;
            D.cost_th_fuel=cost_th_fuel;
            D.cost_th_startstop=cost_th_startstop;
            D.cost_psp_startstop=cost_psp_startstop;
            D.cost_curt=cost_curt;
            D.cost_deficit=cost_deficit;
            D.penTotal=penTotal; D.switchCnt=mode_switch;
        end
    end
end

%% ----------------- GA-PSO 主循环（按你的策略） -----------------
function best = gapsops_optimize(fitfun, nVar, lb, ub, opt)
    popN  = opt.popSize;                                                % 种群规模
    X     = rand(popN, nVar).*(ub'-lb') + lb';                          % 随机初始化位置
    V     = zeros(popN, nVar);                                          % 初始化速度
    Pbest = X; PbestFit = inf(popN,1);                                  % 个体历史最优

    F = zeros(popN,1);
    for i=1:popN, F(i)=fitfun(X(i,:)); end                              % 初评估
    [gbestFit, idx] = min(F); Gbest = X(idx,:);                         % 全局最优
    kmax = opt.maxIterCoarse + opt.maxIterFine;                         % 总迭代次数

    % --------- 粗迭代阶段（M 次） ----------
    for k=1:opt.maxIterCoarse
        w = 1 - 0.2*(k/kmax);                                           % 惯性权重线性递减
        r1 = rand(popN,nVar); r2 = rand(popN,nVar);
        V  = w.*V + opt.c1.*r1.*(Pbest - X) + opt.c2.*r2.*(Gbest - X);  % PSO 速度更新
        X  = X + V;                                                     % 位置更新
        X  = X + opt.v0*(1-0.4*(k/kmax))*(2*rand(popN,nVar)-1);         % 随机变异（扰动）
        X  = max(X, lb'); X = min(X, ub');                              % 越界裁剪

        % 评估与个体/全局最优更新
        for i=1:popN
            Fi = fitfun(X(i,:));
            if Fi < PbestFit(i), Pbest(i,:)=X(i,:); PbestFit(i)=Fi; end
            F(i)=Fi;
        end
        [gbestFitNew, idx] = min(F);
        if gbestFitNew < gbestFit, gbestFit=gbestFitNew; Gbest=X(idx,:); end

        % ---- 精英策略 + GA（选择/交叉/变异） ----
        [F, order] = sort(F,'ascend'); X=X(order,:);                    % 按适应度升序
        elite = X(1,:);                                                 % 保留精英
        P = max(1, round(opt.topReplacePct*popN));                      % 前/后 P%
        X(end-P+1:end,:) = X(1:P,:);                                    % 最差P% ← 最优P%
        V(end-P+1:end,:) = V(end-P+1:end,:) + ...
                           opt.v0*(1-0.4*(k/kmax))*(2*rand(P,nVar)-1);  % 被替换部分再扰动
        midL=P+1; midR=popN-P;
        X(midL:midR,:) = ga_block(X(midL:midR,:), F(midL:midR), opt, lb, ub); % 中段做GA
        X(1,:) = elite;                                                 % 维护精英不变

        % ---- 局部微扰（围绕精英的小步长探索） ----
        q = min(10, popN-2*P); sel = randperm(nVar, min(6,nVar));
        X(P+1:P+q,:) = repmat(elite, q,1);
        X(P+1:P+q, sel) = X(P+1:P+q, sel) + 0.1*opt.v0*(2*rand(q,numel(sel))-1);
        X = max(X, lb'); X = min(X, ub');
    end

    % --------- 细迭代阶段（局部小步搜索） ----------
    for k = opt.maxIterCoarse+1 : kmax
        w = 1 - 0.2*(k/kmax);
        r1 = rand(popN,nVar); r2 = rand(popN,nVar);
        V  = w.*V + 0.6*opt.c1.*r1.*(Pbest - X) + 0.6*opt.c2.*r2.*(Gbest - X); % 减弱学习因子
        step = 0.2*opt.v0;                                              % 小步长
        X = X + V + step*(2*rand(popN,nVar)-1);                         % 细粒度搜索
        X = max(X, lb'); X = min(X, ub');

        evalN = ceil(0.7*popN);                                         % 评估前70%个体
        for i=1:evalN
            Fi = fitfun(X(i,:));
            if Fi<PbestFit(i), Pbest(i,:)=X(i,:); PbestFit(i)=Fi; end
            F(i)=Fi;
        end
        [~, idx] = min(PbestFit); Gbest = Pbest(idx,:); gbestFit = PbestFit(idx);

        % 贪婪局部搜索：在 Gbest 若干维做微扰，择优
        sel = randperm(nVar, randi([1,2]));
        cand = repmat(Gbest, 6, 1);
        cand(:, sel) = cand(:, sel) + 0.05*opt.v0*(2*rand(6,numel(sel))-1);
        cand = max(cand, lb'); cand = min(cand, ub');
        Fcand = arrayfun(@(i) fitfun(cand(i,:)), 1:size(cand,1))';
        [minFc, j] = min(Fcand);
        if minFc < gbestFit, Gbest = cand(j,:); gbestFit = minFc; end
        X(end-size(cand,1)+1:end,:) = cand;                             % 写回尾部
    end
    best = Gbest;                                                       % 返回最优解
end

%% ----------------- GA 子块（选择/交叉/变异） -----------------
function Xnew = ga_block(Xmid, Fmid, opt, lb, ub)
    [m,n] = size(Xmid); Xnew = Xmid;
    rnk   = tiedrank(Fmid);                                            % 适应度排序名次
    selP  = (max(rnk)+1 - rnk); selP = selP/sum(selP);                 % 轮盘赌概率（小优）
    for i=1:2:m-1
        if rand < opt.crossRate                                        % 交叉概率
            p1 = roulette(selP); p2 = roulette(selP);
            alpha  = rand(1,n);                                        % 均匀混合交叉
            child1 = alpha.*Xmid(p1,:) + (1-alpha).*Xmid(p2,:);
            child2 = alpha.*Xmid(p2,:) + (1-alpha).*Xmid(p1,:);
            Xnew(i,:)   = child1; Xnew(i+1,:) = child2;
        end
    end
    mutMask = rand(m,n) < (opt.mutRate/n);                             % 控制总体变异率
    delta   = 0.1*(ub'-lb').*(2*rand(m,n)-1);                          % 变异幅度
    Xnew = Xnew + mutMask.*delta;                                      % 变异
    Xnew = max(Xnew, lb'); Xnew = min(Xnew, ub');                      % 越界裁剪

    function idx = roulette(p)                                         % 轮盘赌选择
        r = rand; c = cumsum(p); idx = find(r<=c,1,'first'); if isempty(idx), idx=1; end
    end
end

%% ----------------- 结果绘图 -----------------
function plot_result(D, Load, PV, Wind, cfg, ps, th)
    t = 1:cfg.T;
    figure('Name','GA-PSO 系统成本最小化结果','Color','w');
    tiledlayout(4,1);

    nexttile;
    plot(t, D.P, 'LineWidth',1.6); grid on;
    yline(ps.Pdis_max,'--'); yline(-ps.Pch_max,'--');
    title('抽蓄净功率 P(t)（放电>0 / 抽水<0）'); xlabel('时段'); ylabel('功率');

    nexttile;
    plot(0:cfg.T, D.E, 'LineWidth',1.6); grid on;
    yline(ps.E_max,'--'); yline(ps.E_min,'--'); yline(D.E(1),':');
    title('库容/等效能量 E(t)'); xlabel('时段'); ylabel('能量');

    nexttile;
    plot(t, D.G, 'LineWidth',1.6); grid on; yline(th.Gmax,'--');
    title('火电出力 G_{th}(t)'); xlabel('时段'); ylabel('功率');

    nexttile;
    S = D.G(:) + PV(:) + Wind(:) + D.P(:);                              % 供应合计
    yyaxis left; plot(t, Load(:), 'LineWidth',1.4); hold on; plot(t, S, 'LineWidth',1.2);
    ylabel('功率');
    yyaxis right; plot(t, D.curt(:), 'LineWidth',1.2); ylabel('弃风弃光');
    grid on; legend('负荷','供应(含PSP)','弃风弃光','Location','best');
    title(sprintf('总成本=%.3f | 燃料=%.3f 启停(热)=%.3f 启停/切换(PSP)=%.3f 弃风弃光=%.3f 罚=%.3f', ...
        D.cost_th_fuel + D.cost_th_startstop + D.cost_psp_startstop + D.cost_curt + D.cost_deficit + D.penTotal, ...
        D.cost_th_fuel, D.cost_th_startstop, D.cost_psp_startstop, D.cost_curt, D.penTotal+D.cost_deficit));
    xlabel('时段');
end

%% ----------------- 工具：带默认值的层级取参 -----------------
function v = getp(s, path, defaultv)
    try
        parts = strsplit(path,'.'); cur = s;
        for i=1:numel(parts)
            if isfield(cur, parts{i}), cur = cur.(parts{i});            % 逐层取字段
            else, v = defaultv; return; end
        end
        v = cur;
    catch
        v = defaultv;                                                   % 取参失败给默认
    end
end

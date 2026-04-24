clear; clc; close all;

%% Load data
data = readtable('caseB_grid_battery_market_hourly.csv');

time = datetime(data.timestamp);
price_MWh = data.day_ahead_price_gbp_per_mwh;   % GBP/MWh
price = price_MWh / 1000;                       % GBP/kWh

T = length(price);
dt = 1;   % hours

%% Battery parameters
Emax = 2000;        % kWh
Pmax = 1000;        % kW
eta_ch = 0.938;
eta_dis = 0.938;
soc0 = 0.5 * Emax;

%% Degradation cost used in main model
throughputPenalty = 0.005;   % GBP/kWh

%% optimisation
n = 3*T;

f = [price*dt + throughputPenalty*dt;
    -price*dt + throughputPenalty*dt;
     zeros(T,1)];

lb = zeros(n,1);

ub = [Pmax*ones(T,1);
      Pmax*ones(T,1);
      Emax*ones(T,1)];

Aeq = zeros(T+1, n);
beq = zeros(T+1, 1);

for t = 1:T
    Aeq(t, t) = -eta_ch * dt;
    Aeq(t, T+t) = (1/eta_dis) * dt;
    Aeq(t, 2*T+t) = 1;

    if t == 1
        beq(t) = soc0;
    else
        Aeq(t, 2*T+t-1) = -1;
    end
end

% End-of-horizon SOC = initial SOC
Aeq(T+1, 2*T+T) = 1;
beq(T+1) = soc0;

options = optimoptions('linprog', 'Display', 'none');

[x, ~, exitflag, output] = linprog(f, [], [], Aeq, beq, lb, ub, options);

if exitflag <= 0
    error('LP optimisation failed.');
end

p_ch_lp = x(1:T);
p_dis_lp = x(T+1:2*T);
soc_lp = [soc0; x(2*T+1:end)];

grossProfit_lp = (p_dis_lp - p_ch_lp) .* price * dt;
degradationCost_lp = throughputPenalty * (p_ch_lp + p_dis_lp) * dt;
netProfit_lp = grossProfit_lp - degradationCost_lp;

%% rule-based model
p25 = prctile(price_MWh, 25);
p75 = prctile(price_MWh, 75);

p_ch_h = zeros(T,1);
p_dis_h = zeros(T,1);
soc_h = zeros(T+1,1);
soc_h(1) = soc0;

for t = 1:T

    if price_MWh(t) <= p25
        maxChargeBySOC = (Emax - soc_h(t)) / (eta_ch * dt);
        p_ch_h(t) = min(Pmax, maxChargeBySOC);

    elseif price_MWh(t) >= p75
        maxDisBySOC = soc_h(t) * eta_dis / dt;
        p_dis_h(t) = min(Pmax, maxDisBySOC);
    end

    soc_h(t+1) = soc_h(t) + eta_ch*p_ch_h(t)*dt - p_dis_h(t)*dt/eta_dis;
end

grossProfit_h = (p_dis_h - p_ch_h) .* price * dt;
degradationCost_h = throughputPenalty * (p_ch_h + p_dis_h) * dt;
netProfit_h = grossProfit_h - degradationCost_h;

%% Verification checks
socBalance_lp = soc_lp(1:end-1) + eta_ch*p_ch_lp*dt - (p_dis_lp*dt)/eta_dis;
socResidual_lp = soc_lp(2:end) - socBalance_lp;

maxSocResidual_lp = max(abs(socResidual_lp));
terminalSocError_lp = soc_lp(end) - soc0;
simultaneousHours_lp = sum((p_ch_lp > 1e-6) & (p_dis_lp > 1e-6));

%% Print summary
fprintf('\n CASE B SUMMARY\n');

fprintf('\n Optimised\n');
fprintf('Gross profit: %.2f GBP\n', sum(grossProfit_lp));
fprintf('Degradation cost: %.2f GBP\n', sum(degradationCost_lp));
fprintf('Net profit: %.2f GBP\n', sum(netProfit_lp));
fprintf('Charge energy: %.2f kWh\n', sum(p_ch_lp*dt));
fprintf('Discharge energy: %.2f kWh\n', sum(p_dis_lp*dt));
fprintf('Throughput: %.2f kWh\n', sum((p_ch_lp+p_dis_lp)*dt));
fprintf('Initial SOC: %.2f kWh\n', soc0);
fprintf('Final SOC: %.2f kWh\n', soc_lp(end));
fprintf('Minimum SOC: %.2f kWh\n', min(soc_lp));
fprintf('Maximum SOC: %.2f kWh\n', max(soc_lp));
fprintf('Max SOC balance residual: %.3e kWh\n', maxSocResidual_lp);
fprintf('Terminal SOC error: %.3e kWh\n', terminalSocError_lp);
fprintf('Simultaneous charge/discharge hours: %d\n', simultaneousHours_lp);
fprintf('Solver iterations: %d\n', output.iterations);

fprintf('\n  Rule-based \n');
fprintf('Low threshold P25: %.2f GBP/MWh\n', p25);
fprintf('High threshold P75: %.2f GBP/MWh\n', p75);
fprintf('Net profit: %.2f GBP\n', sum(netProfit_h));
fprintf('Throughput: %.2f kWh\n', sum((p_ch_h+p_dis_h)*dt));




%% Figure 1: SOC over one representative week
idx = 1:168;

figure;
plot(time(idx), soc_lp(idx), 'LineWidth', 1.8);
grid on;
title('Battery SOC Over 1 Week');
xlabel('Time');
ylabel('SOC (kWh)');
ylim([0 Emax]);
saveas(gcf, 'figure1_soc_one_week.png');

%% Figure 2: Profit: Optimised Model vs Rule-based
figure;
plot(time, cumsum(netProfit_lp), 'LineWidth', 1.8);
hold on;
plot(time, cumsum(netProfit_h), 'LineWidth', 1.8);
grid on;
title('Profit: Optimised Model vs Rule-based');
xlabel('Time');
ylabel('Cumulative net profit (£)');
legend('Optimised', 'Rule-based', 'Location', 'best');
saveas(gcf, 'figure2_cumulative_profit.png');

%% Figure 3: Degradation sensitivity
cDegValues = [0 0.002 0.005 0.01 0.02];
profitVals = zeros(length(cDegValues),1);

for k = 1:length(cDegValues)

    cDeg = cDegValues(k);

    f_deg = [price*dt + cDeg*dt;
            -price*dt + cDeg*dt;
             zeros(T,1)];

    [x_deg, ~, exitflag_deg] = linprog(f_deg, [], [], Aeq, beq, lb, ub, options);

    if exitflag_deg <= 0
        error('LP failed during degradation sensitivity.');
    end

    p_ch = x_deg(1:T);
    p_dis = x_deg(T+1:2*T);

    gross = (p_dis - p_ch) .* price * dt;
    degCost = cDeg * (p_ch + p_dis) * dt;
    net = gross - degCost;

    profitVals(k) = sum(net);
end

figure;

figure;

yyaxis left
h1 = plot(time(idx), price_MWh(idx), 'LineWidth', 1.5);
ylabel('Price (GBP/MWh)');

yyaxis right
h2 = plot(time(idx), soc_lp(idx), 'LineWidth', 1.8);
ylabel('State of Charge (kWh)');
ylim([0 Emax]);

xlabel('Time');
title('Battery Operation: Price and SOC With Degradation');

legend([h1 h2], {'Day-ahead price', 'SOC'}, 'Location', 'best');

grid on;
fprintf('\nFigures saved:\n');
fprintf('figure1_soc_one_week.png\n');
fprintf('figure2_cumulative_profit.png\n');
fprintf('figure3_degradation_profit.png\n');
fprintf('Results saved: caseB_final_results.csv\n');

% fit_diffusion_mle_MS.m
%
% MLE fitting of diffusion_model_ratcliff parameters to monkey
% performance data (multinomial over offset bins -2:+2 + no-stop).
% The no-stop / outside-range category has observed count = 0: every
% no-stop trial the model produces reduces pred_p(1:5), implicitly
% penalising the model for failing to stop within -2:+2.
%
% Free parameters:
%   drift_mean  (v)    - mean of trial-to-trial drift distribution
%   drift_sigma (eta)  - SD of trial-to-trial drift distribution
%   sigma_z            - SD of folded-normal starting-point distribution
%
% All other model parameters are held fixed.

clear; clc;
rng(42)

% =========================================================================
% 1.  Toggle
% =========================================================================
fix_drift = true;   % set to true to fix drift_sigma = 0 (only drift_mean is fitted)

% =========================================================================
% 2.  Fixed model parameters
% =========================================================================
noise    = 0.1;
dt       = 0.01;
step_dur = 1;
bound    = 1;

n_steps_rel = 7;
period      = step_dur;
t_vec       = (0 : dt : n_steps_rel * period)';
n_t         = numel(t_vec);

% =========================================================================
% 3.  Load monkey data and build observed counts
% =========================================================================
targets_monk = [3, 4];
RT_cutoff    = 1100;    % exclude monkey RTs above this value (ms) --> invalid
project_dir  = fileparts(fileparts(mfilename('fullpath')));
monkeys      = {'m1', 'm2'};
monk_offsets = [];

for mk = 1 : numel(monkeys)
    load(fullfile(project_dir, 'data', ...
        sprintf('bhv_data_tbl_%s', monkeys{mk})));
    for sess = 1 : height(dataTable)
        RM = dataTable.RespMat{sess};
        RM(all(RM == 9, 2), :) = [];

        corr_mask    = RM(:,5) == 0;
        err_mask     = RM(:,5) == 6;
        rt_conf      = dataTable.RT_conf{sess};
        rt_err_conf  = dataTable.RT_err_conf{sess};
        corr_counter = cumsum(corr_mask);
        err_counter  = cumsum(err_mask);

        for tg = targets_monk
            tg_mask = RM(:,2) == tg;

            tg_corr_rows = find(corr_mask & tg_mask);
            if ~isempty(tg_corr_rows) && ~isempty(rt_conf)
                rt_c  = rt_conf(corr_counter(tg_corr_rows));
                valid = ~isnan(rt_c) & rt_c <= RT_cutoff;
                monk_offsets = [monk_offsets; zeros(sum(valid), 1)]; 
            end

            tg_err_rows = find(err_mask & tg_mask);
            if ~isempty(tg_err_rows) && ~isempty(rt_err_conf)
                rt_e  = rt_err_conf(err_counter(tg_err_rows));
                off_e = RM(tg_err_rows, 3) - tg;
                valid = ~isnan(rt_e) & rt_e <= RT_cutoff;
                monk_offsets = [monk_offsets; off_e(valid)]; 
            end
        end
    end
end

bins     = -2 : 2;
n_bins   = numel(bins);
n_monk   = numel(monk_offsets);
obs_counts = zeros(1, n_bins + 1);
for b = 1 : n_bins
    obs_counts(b) = sum(monk_offsets == bins(b));
end
obs_counts(end) = 0;       % monkey never fails to stop within -2:+2
n_obs = sum(obs_counts);

fprintf('Loaded %d monkey trials (%d within -2:+2)\n', n_monk, n_obs);
fprintf('Observed proportions  ');
labels = {'-2','-1','0','+1','+2','no-stop'};
for i = 1 : n_bins
    fprintf('%s=%.1f%%  ', labels{i}, 100*obs_counts(i)/n_obs);
end
fprintf('%s=%.1f%%', labels{end}, 0.0);
fprintf('\n\n');

% =========================================================================
% 4.  Optimization bounds and starting values
% =========================================================================
if fix_drift
    %              drift_mean   sigma_z
    x0 = [         0.4,         0.2  ];
    lb = [         0.1,         0.01 ];
    ub = [         1,           1.00 ];
    param_names = {'drift_mean','sigma_z'};
else
    %              drift_mean   drift_sigma   sigma_z
    x0 = [         0.4,        0.10,         0.2  ];
    lb = [         0.25,       0.01,         0.01 ];
    ub = [         0.5,        2.00,         1.00  ];
    param_names = {'drift_mean','drift_sigma','sigma_z'};
end
n_params = numel(x0);

% =========================================================================
% 5.  Number of simulation trials per likelihood evaluation
%
%     Higher n_sim → smoother likelihood surface → better convergence,
%     but slower per evaluation.  Increase to 50 000–100 000 if stalling.
% =========================================================================
n_sim = 20000;

% =========================================================================
% 6.  Multi-start fminsearch with sigmoid transform to enforce bounds
%
%     Bounds are enforced via a logistic (sigmoid) transform:
%       raw (unconstrained) -> bounded:  p = lb + (ub-lb) ./ (1+exp(-r))
%       bounded -> raw:                  r = log((p-lb) ./ (ub-p))
%
%     Multi-start: n_starts random starting points drawn uniformly from
%     the bounded parameter space.  The run with the lowest neg-LL wins.
% =========================================================================
n_starts = 20;

to_raw    = @(p) log((p - lb) ./ (ub - p));
to_params = @(r) lb + (ub - lb) ./ (1 + exp(-r));

obj_raw = @(r) neg_log_lik(to_params(r), fix_drift, bound, obs_counts, n_sim, ...
    n_bins, bins, t_vec, n_t, dt, noise, period);

rng(42)
starts = [x0; lb + (ub - lb) .* rand(n_starts - 1, n_params)];

opts = optimset( ...
    'Display',     'off', ...
    'MaxFunEvals', 600,   ...
    'MaxIter',     300,   ...
    'TolX',        1e-3,  ...
    'TolFun',      1e-3);

fprintf('Multi-start fitting: %d starts, %d params, n_sim=%d...\n\n', ...
    n_starts, n_params, n_sim);

best_nll = Inf;
best_r   = to_raw(x0);
all_nll  = NaN(n_starts, 1);

tic
for s = 1 : n_starts
    r0_s = to_raw(starts(s,:));
    [r_s, nll_s] = fminsearch(obj_raw, r0_s, opts);
    all_nll(s)   = nll_s;
    fprintf('  start %2d/%d  neg-LL = %.2f  [%s]\n', s, n_starts, nll_s, ...
        strjoin(arrayfun(@(v) sprintf('%.3f', v), to_params(r_s), ...
        'UniformOutput', false), '  '));
    if nll_s < best_nll
        best_nll = nll_s;
        best_r   = r_s;
    end
end
t_fit = toc;

x_opt   = to_params(best_r);
nll_opt = best_nll;

fprintf('\n--- Best result (%.0f s total) ---\n', t_fit);
for i = 1 : n_params
    fprintf('  %-14s = %.4f\n', param_names{i}, x_opt(i));
end
fprintf('  neg-LL         = %.2f\n', nll_opt);
fprintf('  neg-LL range   = [%.2f  %.2f]\n\n', min(all_nll), max(all_nll));

% =========================================================================
% 7.  Final model evaluation (larger n_sim for stable comparison)
% =========================================================================
rng(0)
pred_p = simulate_model(x_opt, fix_drift, bound, n_sim * 3, ...
    n_bins, bins, t_vec, n_t, dt, noise, period);

monk_pct = 100 * obs_counts / n_obs;
pred_pct = 100 * pred_p;

fprintf('%-10s', 'Category:');  fprintf('%-9s', labels{:});    fprintf('\n');
fprintf('%-10s', 'Monkey %:');  fprintf('%-9.1f', monk_pct);   fprintf('\n');
fprintf('%-10s', 'Model %:');   fprintf('%-9.1f', pred_pct(1:n_bins)); fprintf('\n');

% =========================================================================
% 8.  Plot
% =========================================================================
lw = 1.5;  fs = 12;
clr_monk  = [1.0 1.0 1.0];
clr_model = [0.4 0.4 0.4];

figure('Color','w','Units','centimeter','Position',[5 5 20 8])

x_ext        = [bins, 3.5];
xlbl_ext     = {'-2','-1','0','+1','+2','NS'};
monk_pct_ext = [monk_pct(1:n_bins), 0];
pred_pct_ext = [pred_pct(1:n_bins), pred_pct(end)];

subplot(1,2,1);  hold on
bar_w = 0.38;
bar(x_ext - bar_w/2, monk_pct_ext, bar_w, ...
    'FaceColor', clr_monk, 'EdgeColor','k','LineWidth',0.8)
bar(x_ext + bar_w/2, pred_pct_ext, bar_w, ...
    'FaceColor', clr_model, 'EdgeColor','k','LineWidth',0.8)
xline(3.0, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'HandleVisibility','off')
legend('Monkey','Model fit','Location','northeast','Box','off')
xticks(x_ext);  xticklabels(xlbl_ext)
xlabel('Offset');  ylabel('Percentage of trials (%)')
title('Performance: monkey vs model')
set(gca,'TickDir','out','LineWidth',lw,'FontSize',fs,'Box','off')

subplot(1,2,2);  hold on
resid = pred_pct_ext - monk_pct_ext;
bar(x_ext, resid, 0.7, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor','k','LineWidth',0.8)
yline(0, '-k', 'LineWidth', 1.5)
xline(3.0, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1)
xticks(x_ext);  xticklabels(xlbl_ext)
xlabel('Offset');  ylabel('\Delta% (model - monkey)')
title('Residuals')
set(gca,'TickDir','out','LineWidth',lw,'FontSize',fs,'Box','off')

if fix_drift
    ttl = sprintf('v=%.3f  \\sigma_z=%.3f  (\\eta fixed=0, \\theta fixed=%.1f)', ...
        x_opt(1), x_opt(2), bound);
else
    ttl = sprintf('v=%.3f  \\eta=%.3f  \\sigma_z=%.3f  (\\theta fixed=%.1f)', ...
        x_opt(1), x_opt(2), x_opt(3), bound);
end
sgtitle(ttl, 'FontSize', fs+1)

% =========================================================================
% Save fitted parameters
% =========================================================================
results_dir = fullfile(project_dir, 'results');
if ~isfolder(results_dir), mkdir(results_dir); end
save(fullfile(results_dir, 'params_perf.mat'), 'x_opt', 'nll_opt', 'fix_drift', 'param_names');
fprintf('Parameters saved to %s\n', fullfile(results_dir, 'params_perf.mat'));

% =========================================================================
% Local functions
% =========================================================================

function nll = neg_log_lik(params, fix_drift, bound, obs_counts, ...
        n_sim, n_bins, bins, t_vec, n_t, dt, noise, period)
    pred_p = simulate_model(params, fix_drift, bound, n_sim, ...
        n_bins, bins, t_vec, n_t, dt, noise, period);
    nll = -sum(obs_counts .* log(pred_p));
end


function pred_p = simulate_model(params, fix_drift, bound, n_trials, ...
        n_bins, bins, t_vec, n_t, dt, noise, period)
    % Returns predicted proportions [p(-2) p(-1) p(0) p(+1) p(+2) p(no-stop)],
    % normalised over all 6 categories.

    drift_mean = params(1);
    if fix_drift
        drift_sigma = 0;
        sigma_z     = max(params(2), 1e-6);
    else
        drift_sigma = max(params(2), 1e-6);
        sigma_z     = max(params(3), 1e-6);
    end

    drift_vals = drift_mean + drift_sigma * randn(1, n_trials);
    dv_curr    = sigma_z * abs(randn(1, n_trials));
    stop_k     = NaN(1, n_trials);
    stopped    = false(1, n_trials);

    for i = 2 : n_t
        active = ~stopped;
        if ~any(active), break; end
        n_act = sum(active);
        dv_curr(active) = max( ...
            dv_curr(active) + drift_vals(active) .* dt + noise * sqrt(dt) * randn(1, n_act), 0);
        k_i = ceil(t_vec(i) / period);
        crossed         = active & (dv_curr >= bound);
        stop_k(crossed) = k_i;
        stopped(crossed) = true;
    end

    offsets = stop_k - 3;

    pred_p = zeros(1, n_bins + 1);
    for b = 1 : n_bins
        pred_p(b) = sum(offsets == bins(b));
    end
    pred_p(end) = sum(isnan(offsets));

    pred_p = max(pred_p, 1e-10);
    pred_p = pred_p / sum(pred_p);
end

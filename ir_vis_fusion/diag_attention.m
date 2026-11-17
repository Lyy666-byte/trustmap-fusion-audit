function diag_attention(nImg)
%DIAG_ATTENTION  Quantify whether the CBAM attention in v4 actually does anything.
%   Evidence for the architectural redesign: if the spatial attention map is
%   near-constant, the module degenerates into a scalar gain (identity-like) and
%   cannot perform any spatial/cross-modal selection.
if nargin < 1 || isempty(nImg), nImg = 30; end
S = load('results/fusion_net_v3.mat'); net = S.net;

d = dir(fullfile('tmp_msrs/test/ir','*.png'));
names = sort({d.name}); names = names(1:min(nImg,numel(names)));
N = numel(names);

sa_mean = zeros(N,1); sa_std = zeros(N,1); sa_cv = zeros(N,1);
sa_p05  = zeros(N,1); sa_p95 = zeros(N,1); sa_frac08 = zeros(N,1);
ca_std  = zeros(N,1); ca_mean = zeros(N,1);
relEff  = zeros(N,1);   % relative change the spatial attention causes on features

for i = 1:N
    nm = names{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end

    [sa,ca,pre] = deal([]);
    sa  = predict(net,X,'Outputs','sa_sig');    % H x W x 1  spatial attention
    ca  = predict(net,X,'Outputs','ca_sig');    % 1 x 1 x 32 channel attention
    pre = predict(net,X,'Outputs','ca_mul');    % features entering spatial att

    sa = double(gather(extractdata(sa)));
    ca = double(gather(extractdata(ca)));
    pre= double(gather(extractdata(pre)));

    sa_mean(i)  = mean(sa(:));
    sa_std(i)   = std(sa(:));
    sa_cv(i)    = sa_std(i)/ (sa_mean(i)+1e-9);
    sa_p05(i)   = prctile(sa(:),5);
    sa_p95(i)   = prctile(sa(:),95);
    sa_frac08(i)= mean(sa(:)>0.8);
    ca_mean(i)  = mean(ca(:));
    ca_std(i)   = std(ca(:));

    % How much does spatial attention change the features, relative to just
    % scaling everything by the mean attention value? (0 => pure scalar gain)
    post   = pre .* sa;
    approx = pre .* sa_mean(i);
    relEff(i) = norm(post(:)-approx(:)) / (norm(post(:))+1e-9);
end

fprintf('\n==== SPATIAL ATTENTION (sa_sig) over %d MSRS test images ====\n',N);
fprintf('mean          : %.4f  (std over images %.4f)\n', mean(sa_mean), std(sa_mean));
fprintf('within-image  : std %.4f | CV %.4f\n', mean(sa_std), mean(sa_cv));
fprintf('p05 / p95     : %.4f / %.4f  (dynamic range %.4f)\n', mean(sa_p05), mean(sa_p95), mean(sa_p95-sa_p05));
fprintf('frac > 0.8    : %.3f\n', mean(sa_frac08));
fprintf('rel. deviation from a pure scalar gain : %.4f\n', mean(relEff));
fprintf('\n==== CHANNEL ATTENTION (ca_sig) ====\n');
fprintf('mean %.4f | std across channels %.4f\n', mean(ca_mean), mean(ca_std));

T = table(names(:), sa_mean, sa_std, sa_cv, sa_p05, sa_p95, sa_frac08, relEff, ...
    'VariableNames',{'image','sa_mean','sa_std','sa_cv','sa_p05','sa_p95','sa_frac_gt08','rel_dev'});
if ~exist('results','dir'), mkdir results; end
writetable(T,'results/diag_attention.csv');
fprintf('\nwrote results/diag_attention.csv\n');
end

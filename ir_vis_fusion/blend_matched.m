function blend_matched(budgetSec, seeds)
%BLEND_MATCHED  Information-matched comparison of our model with every other deep method.
%   For each pair (ours, method k) on MSRS, TNO and RoadScene, the output with the higher dataset-mean
%   MI is blended with the pixel average of the sources, F_a = (1-a) F + a (I_ir + I_vis)/2, using the
%   smallest a at which its dataset-mean MI falls to that of the other output (grid a = 0:0.05:1, then
%   bisection inside the first bracket). Both outputs are then scored with SSIM, Qabf, Nabf, MI and VIF
%   (same functions as metric_full / blend_frontier) and compared with Wilcoxon signed-rank tests.
%   If no a in [0,1] reaches the target the pair is marked unattainable (a = NaN).
%   Our outputs are recomputed for every seed; seed 0 must reproduce the per-image cache (checked).
%   Resumable within a time budget; results/blend_matched.mat, then results/blend_matched.csv.
if nargin < 1, budgetSec = 515; end
if nargin < 2, seeds = 0; end
t0 = tic;
others = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo','MMDRFuse','LUTFuse'};
sets = {'MSRS','tmp_msrs/test/ir','tmp_msrs/test/vi', ...
        {'','results/sota/SeAFusion_MSRS30','results/sota/CDDFuse_MSRS30','results/sota/EMMA_MSRS', ...
         'results/sota/DCEvo_MSRS','results/sota/MMDRFuse_MSRS','results/sota/LUTFuse_MSRS'}; ...
        'TNO','data/TNO/ir','data/TNO/vi', ...
        {'','results/sota/SeAFusion_TNO','results/sota/CDDFuse_TNO','results/sota/EMMA_TNO', ...
         'results/sota/DCEvo_TNO','results/sota/MMDRFuse_TNO','results/sota/LUTFuse_TNO'}; ...
        'RoadScene','data/RoadScene/ir','data/RoadScene/vi', ...
        {'','results/sota/SeAFusion_RoadScene','results/sota/CDDFuse_RoadScene','results/sota/EMMA_RoadScene', ...
         'results/sota/DCEvo_RoadScene','results/sota/MMDRFuse_RoadScene','results/sota/LUTFuse_RoadScene'}};
mnames = {'SSIM','Qabf','Nabf','MI','VIF'};
grid = 0:0.05:1;
af = 'results/blend_matched.mat';
if exist(af,'file'), L = load(af); M = L.M; else, M = struct(); end
Sd = load('results/densefuse.mat'); enc = Sd.enc; dec = Sd.dec;
for s = seeds
    if s == 0, mf = 'results/fusion_net_v3.mat'; else, mf = sprintf('results/model_V7s%d.mat',s); end
    So = load(mf); net = So.net;
    for d = 1:size(sets,1)
        tag = sets{d,1}; key = sprintf('%s_s%d',tag,s);
        P = load(fullfile('results',['peritem_' tag '.mat'])); files = P.files; N = numel(files);
        if isfield(M,key) && all(isfinite(M.(key).done)), continue; end
        if toc(t0) > budgetSec - 150, save(af,'M'); fprintf('PARTIAL (budget)\n'); return; end
        % sources, our fused images and the other methods' fused images (grayscale double)
        IR = cell(1,N); VI = cell(1,N); FO = cell(1,N); FK = cell(numel(others),N);
        for i = 1:N
            ir = imread(fullfile(sets{d,2},files{i})); vi = imread(fullfile(sets{d,3},files{i}));
            IR{i} = gray01(ir); VI{i} = gray01(vi);
            FO{i} = gray01(fuse_images(net,ir,vi));
            for k = 1:numel(others)
                if k == 1, FK{k,i} = gray01(densefuse_fuse(enc,dec,ir,vi));
                else,      FK{k,i} = gray01(imread(fullfile(sets{d,4}{k},files{i}))); end
            end
        end
        if ~isfield(M,key)
            M.(key) = struct('done',nan(1,numel(others)),'a',nan(1,numel(others)),'side',zeros(1,numel(others)), ...
                'ours',nan(numel(others),numel(mnames),N),'other',nan(numel(others),numel(mnames),N), ...
                'native_ours',[],'files',{files});
            M.(key).native_ours = score_all(FO, IR, VI, 0);
            if s == 0   % reproducibility check against the per-image cache of Tables 1-3
                ref = squeeze(P.ACC(strcmp(P.methods,'Ours'),cellfun(@(m) find(strcmp(P.names,m)),mnames),:));
                fprintf('%s seed 0 max |diff| vs peritem: %.2e\n', tag, max(abs(M.(key).native_ours - ref),[],'all'));
            end
            save(af,'M');
        end
        no = M.(key).native_ours;
        for k = 1:numel(others)
            if isfinite(M.(key).done(k)), continue; end
            if toc(t0) > budgetSec - 90, save(af,'M'); fprintf('PARTIAL (budget)\n'); return; end
            nk = squeeze(P.ACC(strcmp(P.methods,others{k}),cellfun(@(m) find(strcmp(P.names,m)),mnames),:));
            miO = mean(no(4,:)); miK = mean(nk(4,:));
            if miO >= miK, Fb = FO; target = miK; side = 1; else, Fb = FK(k,:); target = miO; side = 2; end
            a = match_mi(Fb, IR, VI, target, grid);
            if isnan(a)
                bl = nan(numel(mnames),N);
            else
                bl = score_all(Fb, IR, VI, a);
            end
            if side == 1, M.(key).ours(k,:,:) = bl; M.(key).other(k,:,:) = nk;
            else,         M.(key).ours(k,:,:) = no; M.(key).other(k,:,:) = bl; end
            M.(key).a(k) = a; M.(key).side(k) = side; M.(key).done(k) = 1;
            save(af,'M');
            fprintf('%s %-9s side %d a %.4f MI %.4f -> %.4f\n', key, others{k}, side, a, max(miO,miK), target);
        end
    end
end
% ---- write the summary table
fid = fopen('results/blend_matched.csv','w');
hdr = 'dataset,seed,method,blended,a,MI_ours,MI_other';
for m = [1 2 3 5], hdr = sprintf('%s,%s_ours,%s_other,%s_p',hdr,mnames{m},mnames{m},mnames{m}); end
fprintf(fid,'%s\n',hdr);
for key = fieldnames(M)'
    Z = M.(key{1}); parts = split(key{1},'_s'); tag = parts{1}; s = str2double(parts{2});
    for k = 1:numel(others)
        sideName = {'ours','other'}; sideName = sideName{Z.side(k)};
        o = squeeze(Z.ours(k,:,:)); q = squeeze(Z.other(k,:,:));
        row = sprintf('%s,%d,%s,%s,%.4f,%.6f,%.6f',tag,s,others{k},sideName,Z.a(k),mean(o(4,:)),mean(q(4,:)));
        for m = [1 2 3 5]
            if isnan(Z.a(k)), p = NaN; else, p = signrank(o(m,:),q(m,:)); end
            row = sprintf('%s,%.6f,%.6f,%.3g',row,mean(o(m,:)),mean(q(m,:)),p);
        end
        fprintf(fid,'%s\n',row);
    end
end
fclose(fid);
type('results/blend_matched.csv');
fprintf('FINALIZED blend_matched\n');
end

% -------------------------------------------------------------------------------------------------
function g = gray01(x)
g = im2double(x); if size(g,3) > 1, g = rgb2gray(g); end
end

function V = score_all(F, IR, VI, a)
N = numel(F); V = nan(5,N);
for i = 1:N
    irg = IR{i}; vig = VI{i}; Fa = im2double(im2uint8((1-a)*F{i} + a*(irg+vig)/2));   % 8-bit, as in metric_full
    V(:,i) = [(ssim(Fa,irg)+ssim(Fa,vig))/2; analysis_Qabf(irg,vig,Fa); analysis_nabf(Fa,irg,vig); ...
              analysis_MI(im2uint8(irg),im2uint8(vig),im2uint8(Fa)); vifp(irg*255,Fa*255)+vifp(vig*255,Fa*255)];
end
end

function m = mean_mi(F, IR, VI, a)
m = 0;
for i = 1:numel(F)
    Fa = (1-a)*F{i} + a*(IR{i}+VI{i})/2;
    m = m + analysis_MI(im2uint8(IR{i}),im2uint8(VI{i}),im2uint8(Fa));
end
m = m/numel(F);
end

function a = match_mi(F, IR, VI, target, grid)
% smallest a with mean MI <= target: first grid bracket, then bisection
prev = grid(1); a = NaN;
for g = grid(2:end)
    if mean_mi(F, IR, VI, g) <= target
        lo = prev; hi = g;
        for it = 1:14
            mid = (lo+hi)/2;
            if mean_mi(F, IR, VI, mid) <= target, hi = mid; else, lo = mid; end
        end
        a = hi; return;
    end
    prev = g;
end
end

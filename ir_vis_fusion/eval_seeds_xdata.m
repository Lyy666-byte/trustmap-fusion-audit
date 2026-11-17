function eval_seeds_xdata(budgetSec)
%EVAL_SEEDS_XDATA  Seeds 1 and 2 of the full model on the three test sets, compared with the
%   competitors. Seed 0 (the reported model) and the competitors are read from the main
%   per-image caches results/peritem_<tag>.mat (same images, same metric code). Seeds 1-2 on
%   MSRS are read from results/seeds_acc.mat (eval_seeds); TNO and RoadScene are computed here.
%   Resumable; per-image results in results/seeds_xdata_acc.mat. Writes results/seeds_xdata.csv:
%   for every dataset, seed and metric, the value, the rank among the six deep methods and the
%   paired p-value against the best other deep method.
if nargin < 1, budgetSec = 480; end
t0 = tic;
mods = {'results/model_V7s1.mat','results/model_V7s2.mat'};
sets = {'TNO','data/TNO/ir','data/TNO/vi'; 'RoadScene','data/RoadScene/ir','data/RoadScene/vi'};
nets = cell(1,2); for k = 1:2, S = load(mods{k}); nets{k} = S.net; end
af = 'results/seeds_xdata_acc.mat';
if exist(af,'file'), L = load(af); R = L.R; else, R = struct(); end
for d = 1:size(sets,1)
    tag = sets{d,1};
    P = load(fullfile('results',['peritem_' tag '.mat'])); files = P.files; N = numel(files); K = numel(P.names);
    if ~isfield(R,tag), R.(tag) = nan(2,K,N); end
    for i = 1:N
        if ~any(isnan(R.(tag)(:,:,i)),'all'), continue; end
        if toc(t0) > budgetSec, save(af,'R'); fprintf('PARTIAL (budget)\n'); return; end
        nm = files{i};
        ir = imread(fullfile(sets{d,2},nm)); vi = imread(fullfile(sets{d,3},nm));
        irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
        vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
        for k = 1:2
            R.(tag)(k,:,i) = metric_full(fuse_images(nets{k},ir,vi),irg,vig);
        end
        save(af,'R');
        fprintf('%s %2d/%d %s\n',tag,i,N,nm);
    end
end

deep = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo'};          % general-purpose deep methods
light = {'MMDRFuse','LUTFuse'};                                      % dedicated lightweight networks
lower = {'Nabf'};
S0 = load('results/seeds_acc.mat');
fid = fopen('results/seeds_xdata.csv','w');
fprintf(fid,'dataset,seed,metric,value,rank_among_6_deep,best_other_deep,best_other_value,p_vs_best_other,rank_among_8_incl_lightweight\n');
for tagc = {'MSRS','TNO','RoadScene'}
    tag = tagc{1};
    P = load(fullfile('results',['peritem_' tag '.mat']));
    seedsAcc = {P.ACC(strcmp(P.methods,'Ours'),:,:)};
    if strcmp(tag,'MSRS'), seedsAcc{2} = S0.ACC(1,:,:); seedsAcc{3} = S0.ACC(2,:,:);
    else, seedsAcc{2} = R.(tag)(1,:,:); seedsAcc{3} = R.(tag)(2,:,:); end
    for s = 1:3
        for mm = {'SSIM','Qabf','Nabf','VIF','MI','EN'}
            j = find(strcmp(P.names,mm{1})); low = any(strcmp(lower,mm{1}));
            ours = squeeze(seedsAcc{s}(1,j,:));
            vals = zeros(1,numel(deep)); for q = 1:numel(deep), vals(q) = mean(P.ACC(strcmp(P.methods,deep{q}),j,:)); end
            if low, [bv,bq] = min(vals); rk = 1 + sum(vals < mean(ours));
            else,   [bv,bq] = max(vals); rk = 1 + sum(vals > mean(ours)); end
            other = squeeze(P.ACC(strcmp(P.methods,deep{bq}),j,:));
            v8 = vals; for q = 1:numel(light), v8(end+1) = mean(P.ACC(strcmp(P.methods,light{q}),j,:)); end %#ok<AGROW>
            if low, rk8 = 1 + sum(v8 < mean(ours)); else, rk8 = 1 + sum(v8 > mean(ours)); end
            fprintf(fid,'%s,%d,%s,%.4f,%d,%s,%.4f,%.3g,%d\n',tag,s-1,mm{1},mean(ours),rk,deep{bq},bv,signrank(ours,other),rk8);
        end
    end
end
fclose(fid);
% every seed against every other deep method (paired over the images), for the claims in the text
fid = fopen('results/seeds_xdata_pairs.csv','w');
fprintf(fid,'dataset,seed,metric,ours,method,other,p_signrank\n');
for tagc = {'MSRS','TNO','RoadScene'}
    tag = tagc{1};
    P = load(fullfile('results',['peritem_' tag '.mat']));
    seedsAcc = {P.ACC(strcmp(P.methods,'Ours'),:,:)};
    if strcmp(tag,'MSRS'), seedsAcc{2} = S0.ACC(1,:,:); seedsAcc{3} = S0.ACC(2,:,:);
    else, seedsAcc{2} = R.(tag)(1,:,:); seedsAcc{3} = R.(tag)(2,:,:); end
    for s = 1:3
        for mm = {'SSIM','Qabf','Nabf','EN','MI','VIF','SD','CC'}
            j = find(strcmp(P.names,mm{1}));
            ours = squeeze(seedsAcc{s}(1,j,:));
            allm = [deep light];
            for q = 1:numel(allm)
                other = squeeze(P.ACC(strcmp(P.methods,allm{q}),j,:));
                fprintf(fid,'%s,%d,%s,%.6f,%s,%.6f,%.3g\n',tag,s-1,mm{1},mean(ours),allm{q},mean(other),signrank(ours,other));
            end
        end
    end
end
fclose(fid);
type('results/seeds_xdata.csv');
fprintf('FINALIZED eval_seeds_xdata\n');
end

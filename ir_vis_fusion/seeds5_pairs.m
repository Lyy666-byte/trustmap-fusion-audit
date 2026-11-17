function seeds5_pairs()
%SEEDS5_PAIRS  Pairwise comparison of every seed of the full model (A3, seeds 0-4) with every other deep
%   method on MSRS, TNO and RoadScene: mean of ours, mean of the other method, Wilcoxon signed-rank p
%   (same per-image values as seeds5_xdata). results/seeds5_pairs.csv.
B = load('results/abl2_acc.mat'); S1 = load('results/seeds_acc.mat'); Q = load('results/seedq_acc.mat');
A3 = cat(1, B.ACC(strcmp(B.labs,'A3_diff_trust'),:,:), S1.ACC(1:2,:,:), Q.ACC(4:5,:,:));
X = load('results/seeds_xdata_acc.mat'); L = load('results/seeds5_acc.mat'); C = L.C;
others = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo','MMDRFuse','LUTFuse'};
fid = fopen('results/seeds5_pairs.csv','w');
fprintf(fid,'dataset,seed,metric,ours,method,other,p\n');
for tagc = {'MSRS','TNO','RoadScene'}
    tag = tagc{1}; P = load(fullfile('results',['peritem_' tag '.mat']));
    if strcmp(tag,'MSRS'), sd = {A3(1,:,:),A3(2,:,:),A3(3,:,:),A3(4,:,:),A3(5,:,:)};
    else, sd = {P.ACC(strcmp(P.methods,'Ours'),:,:), X.R.(tag)(1,:,:), X.R.(tag)(2,:,:), C.(tag)(1,:,:), C.(tag)(2,:,:)}; end
    for s = 1:5
        for mm = {'SSIM','Qabf','Nabf','VIF','MI'}
            j = find(strcmp(P.names,mm{1})); ours = squeeze(sd{s}(1,j,:));
            for k = 1:numel(others)
                o = squeeze(P.ACC(strcmp(P.methods,others{k}),j,:));
                fprintf(fid,'%s,%d,%s,%.6f,%s,%.6f,%.3g\n',tag,s-1,mm{1},mean(ours),others{k},mean(o),signrank(ours,o));
            end
        end
    end
end
fclose(fid);
fprintf('FINALIZED seeds5_pairs\n');
end

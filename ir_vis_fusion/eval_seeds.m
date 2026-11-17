function eval_seeds(budgetSec)
%EVAL_SEEDS  Training-seed variability of the full model (A3) and the trust-free ablation (A1).
%   Seed 0 = the models reported in the paper (fusion_net_v3.mat = A3, model_AB1.mat = A1; their
%   per-image metrics are read from results/abl2_acc.mat). Seeds 1 and 2 were trained with
%   train_resume(...,seed) on identical data, loss and schedule (models V7s1, V7s2, AB1s1, AB1s2).
%   All are evaluated on the MSRS30 subset with metric_full (13 metrics, full resolution).
%   Resumable within a time budget; per-image results in results/seeds_acc.mat.
%   When complete it also runs, for every A3 seed, the constant-map intervention
%   (test_constant_w) and the person/car-vs-background trust-map ratio on the MSRS labels, and
%   writes results/seeds_metrics.csv, results/seeds_a3_vs_a1.csv and results/seeds_trustmap.csv.
if nargin < 1, budgetSec = 480; end
t0 = tic;
mods = {'results/model_V7s1.mat','results/model_V7s2.mat','results/model_AB1s1.mat','results/model_AB1s2.mat'};
tags = {'A3_s1','A3_s2','A1_s1','A1_s2'};
names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};
L0 = load('results/peritem_MSRS.mat'); files = L0.files; N = numel(files);
nets = cell(1,4); for k = 1:4, S = load(mods{k}); nets{k} = S.net; end
af = 'results/seeds_acc.mat';
if exist(af,'file'), L = load(af); ACC = L.ACC; DEV = L.DEV;
else, ACC = nan(4,numel(names),N); DEV = nan(4,N); end
for i = 1:N
    if ~any(isnan(ACC(:,:,i)),'all'), continue; end
    if toc(t0) > budgetSec, save(af,'ACC','DEV','tags','names','files'); fprintf('PARTIAL (budget)\n'); return; end
    nm = files{i};
    ir = imread(fullfile('tmp_msrs/test/ir',nm)); vi = imread(fullfile('tmp_msrs/test/vi',nm));
    irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
    vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
    for k = 1:4
        F = fuse_images(nets{k},ir,vi);
        ACC(k,:,i) = metric_full(F,irg,vig);
        if any(strcmp({nets{k}.Layers.Name},'trust'))
            X = dlarray(reshape(single(cat(3,im2single(irg),im2single(vig))),[size(irg,1) size(irg,2) 2 1]),'SSCB');
            if canUseGPU, X = gpuArray(X); end
            W = double(gather(extractdata(predict(nets{k},X,'Outputs','trust'))));
            DEV(k,i) = norm(W(:)-mean(W(:)))/(norm(W(:))+1e-9);
        end
    end
    save(af,'ACC','DEV','tags','names','files');
    fprintf('seeds %2d/%d %s\n',i,N,nm);
end

% ---- seed 0 from the ablation cache (same images, same metric code)
B = load('results/abl2_acc.mat');
A3 = cat(1, B.ACC(strcmp(B.labs,'A3_diff_trust'),:,:), ACC(1:2,:,:));     % 3 x K x N
A1 = cat(1, B.ACC(strcmp(B.labs,'A1_dual_cat'),:,:),   ACC(3:4,:,:));
devA3 = [mean(B.DEV(strcmp(B.labs,'A3_diff_trust'),:),'omitnan'), mean(DEV(1,:)), mean(DEV(2,:))];

fid = fopen('results/seeds_metrics.csv','w');
fprintf(fid,'config,seed,%s,trust_dev\n',strjoin(names,','));
for s = 1:3
    fprintf(fid,'A3,%d,%s,%.6f\n',s-1,strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(A3(s,:,:),3),'uni',0),','),devA3(s));
end
for s = 1:3
    fprintf(fid,'A1,%d,%s,\n',s-1,strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(A1(s,:,:),3),'uni',0),','));
end
mA3 = mean(A3,3); mA1 = mean(A1,3);
fprintf(fid,'A3,mean,%s,%.6f\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(mA3,1),'uni',0),','),mean(devA3));
fprintf(fid,'A3,std,%s,%.6f\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),std(mA3,0,1),'uni',0),','),std(devA3));
fprintf(fid,'A1,mean,%s,\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(mA1,1),'uni',0),','));
fprintf(fid,'A1,std,%s,\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),std(mA1,0,1),'uni',0),','));
fclose(fid);

% ---- A3 vs A1 for each seed pair (paired over the 30 images) and seed-averaged per-image values
fid = fopen('results/seeds_a3_vs_a1.csv','w');
fprintf(fid,'metric,diff_s0,p_s0,diff_s1,p_s1,diff_s2,p_s2,diff_seedavg,p_seedavg\n');
for j = 1:numel(names)
    row = names{j};
    for s = 1:3
        a = squeeze(A3(s,j,:)); b = squeeze(A1(s,j,:));
        row = sprintf('%s,%.4f,%.3g',row,mean(a-b),signrank(a,b));
    end
    a = squeeze(mean(A3(:,j,:),1)); b = squeeze(mean(A1(:,j,:),1));
    fprintf(fid,'%s,%.4f,%.3g\n',row,mean(a-b),signrank(a,b));
end
fclose(fid);

% ---- trust map of every A3 seed: constant-map test and person/car-vs-background ratio
allm = {'results/fusion_net_v3.mat','results/model_V7s1.mat','results/model_V7s2.mat'};
fid = fopen('results/seeds_trustmap.csv','w');
fprintf(fid,'seed,loss_real,loss_meanconst,rise_pct,n_increase,p,person_ratio,p_person,car_ratio,p_car\n');
for s = 1:3
    res = test_constant_w(30, allm{s}, sprintf('results/constant_w_test_seed%d.csv',s-1));
    S = load(allm{s});
    [pr,pp,cr,pc] = classRatio(S.net, files);
    fprintf(fid,'%d,%.4f,%.4f,%.2f,%d,%.3g,%.3f,%.3g,%.3f,%.3g\n',s-1,mean(res(:,1)),mean(res(:,2)), ...
        100*(mean(res(:,2))-mean(res(:,1)))/mean(res(:,1)),sum(res(:,2)>res(:,1)),signrank(res(:,2),res(:,1)), ...
        pr,pp,cr,pc);
end
fclose(fid);
type('results/seeds_metrics.csv'); type('results/seeds_a3_vs_a1.csv'); type('results/seeds_trustmap.csv');
fprintf('FINALIZED eval_seeds\n');
end

function [pr,pp,cr,pc] = classRatio(net, files)
% mean W on person / car pixels over mean W on unlabelled pixels of the same images (>= 50 px)
P = []; C = [];
for i = 1:numel(files)
    nm = files{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm))); if size(ir,3)>1, ir = rgb2gray(ir); end
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm))); if size(vi,3)>1, vi = rgb2gray(vi); end
    lab = imread(fullfile('tmp_msrs/test/Segmentation_labels',nm));
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end
    W = double(gather(extractdata(predict(net,X,'Outputs','trust'))));
    bg = mean(W(lab==0));
    if nnz(lab==2) >= 50, P(end+1,:) = [mean(W(lab==2)) bg]; end %#ok<AGROW>
    if nnz(lab==1) >= 50, C(end+1,:) = [mean(W(lab==1)) bg]; end %#ok<AGROW>
end
pr = mean(P(:,1))/mean(P(:,2)); pp = signrank(P(:,1),P(:,2));
cr = mean(C(:,1))/mean(C(:,2)); pc = signrank(C(:,1),C(:,2));
end

function seedq(budgetSec)
%SEEDQ  Seeds 3 and 4 of V7 (reported architecture) and V8 (scale-anchored), plus the mean trust-map
%   level of every model, so that seed-to-seed variability can be compared over five seeds each.
%   Same data, loss, schedule and chunking as all other runs (train_resume with seed > 0 re-seeds at
%   the resume). One 20-epoch chunk per call (only if it fits), then MSRS30 evaluation (metric_full,
%   mean W, deviation of W from a constant) with the remaining time; results/seedq_acc.mat.
%   When complete: results/seeds5_v7_v8.csv (per model) and results/seeds5_summary.csv (mean, std,
%   Brown-Forsythe test of equal variances V7 vs V8 per metric).
if nargin < 1, budgetSec = 515; end
t0 = tic;
train = {'V7s3','build_fusion_net_v7',3; 'V7s4','build_fusion_net_v7',4; ...
         'V8s3','build_fusion_net_v8',3; 'V8s4','build_fusion_net_v8',4};
for j = 1:size(train,1) * (budgetSec >= 360)
    if exist(fullfile('results',['model_' train{j,1} '.mat']),'file'), continue; end
    train_resume(train{j,1},train{j,2},'fusion_loss_v5',40,200,20,20,train{j,3});
    break
end
% all ten models: V7 s0..s4, V8 s0..s4
mods = {'results/fusion_net_v3.mat','results/model_V7s1.mat','results/model_V7s2.mat', ...
        'results/model_V7s3.mat','results/model_V7s4.mat', ...
        'results/model_V8s0.mat','results/model_V8s1.mat','results/model_V8s2.mat', ...
        'results/model_V8s3.mat','results/model_V8s4.mat'};
arch = [7 7 7 7 7 8 8 8 8 8]; seed = [0 1 2 3 4 0 1 2 3 4];
names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};
L0 = load('results/peritem_MSRS.mat'); files = L0.files; N = numel(files);
af = 'results/seedq_acc.mat';
if exist(af,'file'), L = load(af); ACC = L.ACC; DEV = L.DEV; MW = L.MW;
else
    ACC = nan(10,numel(names),N); DEV = nan(10,N); MW = nan(10,N);
    % reuse the metrics already computed for the existing models (same images, same code)
    B = load('results/abl2_acc.mat'); S1 = load('results/seeds_acc.mat'); V8 = load('results/v8_acc.mat');
    ACC(1,:,:) = B.ACC(strcmp(B.labs,'A3_diff_trust'),:,:); ACC(2:3,:,:) = S1.ACC(1:2,:,:);
    ACC(6:8,:,:) = V8.ACC; DEV(6:8,:) = V8.DEV; MW(6:8,:) = V8.MW;
end
for j = 1:10
    if ~exist(mods{j},'file'), continue; end
    needM = any(isnan(ACC(j,:,:)),'all'); needW = any(isnan(MW(j,:)));
    if ~needM && ~needW, continue; end
    S = load(mods{j}); net = S.net;
    for i = 1:N
        if ~any(isnan(ACC(j,:,i))) && ~isnan(MW(j,i)), continue; end
        if toc(t0) > budgetSec, save(af,'ACC','DEV','MW','mods','names','files'); fprintf('PARTIAL (budget)\n'); return; end
        nm = files{i};
        ir = imread(fullfile('tmp_msrs/test/ir',nm)); vi = imread(fullfile('tmp_msrs/test/vi',nm));
        irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
        vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
        if any(isnan(ACC(j,:,i))), ACC(j,:,i) = metric_full(fuse_images(net,ir,vi),irg,vig); end
        X = dlarray(reshape(single(cat(3,im2single(irg),im2single(vig))),[size(irg,1) size(irg,2) 2 1]),'SSCB');
        if canUseGPU, X = gpuArray(X); end
        W = double(gather(extractdata(predict(net,X,'Outputs','trust'))));
        MW(j,i) = mean(W(:)); DEV(j,i) = norm(W(:)-mean(W(:)))/(norm(W(:))+1e-9);
        save(af,'ACC','DEV','MW','mods','names','files');
    end
    fprintf('evaluated %s\n', mods{j});
end
if any(isnan(ACC(:))) || any(isnan(MW(:))), fprintf('SEEDQ not finished\n'); return; end
fid = fopen('results/seeds5_v7_v8.csv','w');
fprintf(fid,'arch,seed,%s,meanW,trust_dev\n',strjoin(names,','));
for j = 1:10
    fprintf(fid,'V%d,%d,%s,%.6f,%.6f\n',arch(j),seed(j),strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(ACC(j,:,:),3),'uni',0),','),mean(MW(j,:)),mean(DEV(j,:)));
end
fclose(fid);
M = [mean(ACC,3), mean(MW,2), mean(DEV,2)]; cols = [names, {'meanW','trust_dev'}];
fid = fopen('results/seeds5_summary.csv','w');
fprintf(fid,'quantity,V7_mean,V7_std,V8_mean,V8_std,std_ratio,p_BrownForsythe,p_ranksum_means\n');
for c = 1:numel(cols)
    a = M(arch==7,c); b = M(arch==8,c);
    pv = vartestn([a;b],[ones(5,1);2*ones(5,1)],'TestType','BrownForsythe','Display','off');
    fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f,%.3f,%.3g,%.3g\n',cols{c},mean(a),std(a),mean(b),std(b),std(a)/max(std(b),eps),pv,ranksum(a,b));
end
fclose(fid);
type('results/seeds5_v7_v8.csv'); type('results/seeds5_summary.csv');
fprintf('FINALIZED seedq\n');
end

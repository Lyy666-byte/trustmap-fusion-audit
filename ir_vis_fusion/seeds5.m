function seeds5(budgetSec)
%SEEDS5  Extends the training-seed study from three to five seeds (seeds 3 and 4).
%   Full model (A3 = V7): models V7s3, V7s4 (trained by seedq.m; MSRS30 metrics in seedq_acc.mat).
%   Trust-free ablation (A1): AB1s3, AB1s4 are trained here (same data, loss, schedule, chunking).
%   Seeds 3 and 4 of A3 are evaluated on TNO and RoadScene, under the three degradations of
%   Section 4.9, and with the constant-map test and the person/car trust ratio, exactly as seeds 1-2.
%   Resumable within a time budget; caches results/seeds5_acc.mat. When complete it writes
%   results/seeds5_metrics.csv, seeds5_a3_vs_a1.csv, seeds5_trustmap.csv, seeds5_xdata.csv,
%   seeds5_robust.csv.
if nargin < 1, budgetSec = 515; end
t0 = tic;
% ---- 1. training (one 20-epoch chunk per call, only if it fits)
tr = {'AB1s3',3; 'AB1s4',4};
for j = 1:2 * (budgetSec >= 360)
    if exist(fullfile('results',['model_' tr{j,1} '.mat']),'file'), continue; end
    train_resume(tr{j,1},'build_abl','fusion_loss_v5',40,200,20,'A1_dual_cat',tr{j,2});
    break
end
names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};
af = 'results/seeds5_acc.mat';
if exist(af,'file'), C = load(af); C = C.C; else, C = struct(); end
P = load('results/peritem_MSRS.mat'); files = P.files;
% ---- 2. MSRS30 metrics of A1 seeds 3-4
if ~isfield(C,'A1'), C.A1 = nan(2,numel(names),numel(files)); end
for j = 1:2
    mf = fullfile('results',['model_' tr{j,1} '.mat']);
    if ~exist(mf,'file'), continue; end
    S = load(mf); net = S.net;
    for i = 1:numel(files)
        if ~any(isnan(C.A1(j,:,i))), continue; end
        if toc(t0) > budgetSec, save(af,'C'); fprintf('PARTIAL (budget)\n'); return; end
        [ir,vi,irg,vig] = readpair('tmp_msrs/test/ir','tmp_msrs/test/vi',files{i});
        C.A1(j,:,i) = metric_full(fuse_images(net,ir,vi),irg,vig);
        save(af,'C');
    end
end
% ---- 3. TNO and RoadScene for A3 seeds 3-4
m34 = {'results/model_V7s3.mat','results/model_V7s4.mat'};
nets = cell(1,2); for j = 1:2, S = load(m34{j}); nets{j} = S.net; end
xs = {'TNO','data/TNO/ir','data/TNO/vi'; 'RoadScene','data/RoadScene/ir','data/RoadScene/vi'};
for d = 1:2
    tag = xs{d,1}; Pd = load(fullfile('results',['peritem_' tag '.mat'])); fd = Pd.files;
    if ~isfield(C,tag), C.(tag) = nan(2,numel(names),numel(fd)); end
    for i = 1:numel(fd)
        if ~any(isnan(C.(tag)(:,:,i)),'all'), continue; end
        if toc(t0) > budgetSec, save(af,'C'); fprintf('PARTIAL (budget)\n'); return; end
        [ir,vi,irg,vig] = readpair(xs{d,2},xs{d,3},fd{i});
        for j = 1:2, C.(tag)(j,:,i) = metric_full(fuse_images(nets{j},ir,vi),irg,vig); end
        save(af,'C');
    end
end
% ---- 4. robustness for A3 seeds 3-4 (same degraded inputs, clean references)
RA = load('results/robust_acc.mat');
if ~isfield(C,'robust'), C.robust = nan(numel(RA.conds),2,numel(names),numel(RA.files)); end
for c = 1:numel(RA.conds)
    dd = fullfile('results','robust',RA.conds{c});
    for i = 1:numel(RA.files)
        if ~any(isnan(C.robust(c,:,:,i)),'all'), continue; end
        if toc(t0) > budgetSec, save(af,'C'); fprintf('PARTIAL (budget)\n'); return; end
        nm = RA.files{i};
        [~,~,irc,vic] = readpair('tmp_msrs/test/ir','tmp_msrs/test/vi',nm);
        ir = imread(fullfile(dd,'ir',nm)); vi = imread(fullfile(dd,'vi',nm));
        for j = 1:2, C.robust(c,j,:,i) = metric_full(fuse_images(nets{j},ir,vi),irc,vic); end
        save(af,'C');
    end
end
% ---- 5. constant-map test and person/car ratio for A3 seeds 3-4
if ~isfield(C,'tm')
    if budgetSec - toc(t0) < 200, save(af,'C'); fprintf('PARTIAL (budget)\n'); return; end
    C.tm = nan(2,7);
    for j = 1:2
        res = test_constant_w(30, m34{j}, sprintf('results/constant_w_test_seed%d.csv', j+2));
        [pr,pp,cr,pc] = classRatio(nets{j}, files);
        C.tm(j,:) = [mean(res(:,1)) mean(res(:,2)) sum(res(:,2)>res(:,1)) signrank(res(:,2),res(:,1)) pr pp cr];
        C.tm_pc(j) = pc;
    end
    save(af,'C');
end
if ~exist('results/model_AB1s4.mat','file') || any(isnan(C.A1(:))), fprintf('SEEDS5 not finished\n'); return; end
summarize(C, names, files);
fprintf('FINALIZED seeds5\n');
end

% -------------------------------------------------------------------------------------------------
function summarize(C, names, files)
B = load('results/abl2_acc.mat'); S1 = load('results/seeds_acc.mat'); Q = load('results/seedq_acc.mat');
A3 = cat(1, B.ACC(strcmp(B.labs,'A3_diff_trust'),:,:), S1.ACC(1:2,:,:), Q.ACC(4:5,:,:));   % seeds 0-4
A1 = cat(1, B.ACC(strcmp(B.labs,'A1_dual_cat'),:,:),   S1.ACC(3:4,:,:), C.A1);
devA3 = mean(Q.DEV(1:5,:),2)';  mwA3 = mean(Q.MW(1:5,:),2)';
TM = readtable('results/seeds_trustmap.csv');
rise = [TM.rise_pct' , 100*(C.tm(:,2)-C.tm(:,1))'./C.tm(:,1)'];
ninc = [TM.n_increase', C.tm(:,3)']; pr = [TM.person_ratio', C.tm(:,5)']; cr = [TM.car_ratio', C.tm(:,7)'];
pp = [TM.p_person', C.tm(:,6)'];
fid = fopen('results/seeds5_metrics.csv','w');
fprintf(fid,'config,seed,%s,trust_dev,meanW,loss_rise_pct,n_increase,person_ratio,p_person,car_ratio\n',strjoin(names,','));
for s = 1:5
    fprintf(fid,'A3,%d,%s,%.6f,%.6f,%.2f,%d,%.4f,%.3g,%.4f\n',s-1,strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(A3(s,:,:),3),'uni',0),','), ...
        devA3(s),mwA3(s),rise(s),ninc(s),pr(s),pp(s),cr(s));
end
for s = 1:5
    fprintf(fid,'A1,%d,%s,,,,,,,\n',s-1,strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(A1(s,:,:),3),'uni',0),','));
end
m3 = mean(A3,3); m1 = mean(A1,3);
fprintf(fid,'A3,mean,%s,%.6f,%.6f,%.2f,,%.4f,,%.4f\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(m3,1),'uni',0),','),mean(devA3),mean(mwA3),mean(rise),mean(pr),mean(cr));
fprintf(fid,'A3,std,%s,%.6f,%.6f,%.2f,,%.4f,,%.4f\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),std(m3,0,1),'uni',0),','),std(devA3),std(mwA3),std(rise),std(pr),std(cr));
fprintf(fid,'A1,mean,%s,,,,,,,\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),mean(m1,1),'uni',0),','));
fprintf(fid,'A1,std,%s,,,,,,,\n',strjoin(arrayfun(@(x) sprintf('%.6f',x),std(m1,0,1),'uni',0),','));
fclose(fid);
fid = fopen('results/seeds5_a3_vs_a1.csv','w');
fprintf(fid,'metric,%s,diff_seedavg,p_seedavg,n_seeds_A3_higher\n',strjoin(arrayfun(@(s) sprintf('diff_s%d,p_s%d',s,s),0:4,'uni',0),','));
for j = 1:numel(names)
    row = names{j};
    for s = 1:5
        a = squeeze(A3(s,j,:)); b = squeeze(A1(s,j,:));
        row = sprintf('%s,%.6f,%.3g',row,mean(a-b),signrank(a,b));
    end
    a = squeeze(mean(A3(:,j,:),1)); b = squeeze(mean(A1(:,j,:),1));
    fprintf(fid,'%s,%.6f,%.3g,%d\n',row,mean(a-b),signrank(a,b),sum(m3(:,j) > m1(:,j)));
end
fclose(fid);
% ranks among the 6 general-purpose deep methods and among all 8, for seeds 0-4
deep = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo'}; light = {'MMDRFuse','LUTFuse'};
X = load('results/seeds_xdata_acc.mat');
fid = fopen('results/seeds5_xdata.csv','w');
fprintf(fid,'dataset,seed,metric,value,rank6,best_other6,best_value6,p_vs_best6,rank8,MMDRFuse,p_vs_MMDRFuse,LUTFuse,p_vs_LUTFuse\n');
for tagc = {'MSRS','TNO','RoadScene'}
    tag = tagc{1}; P = load(fullfile('results',['peritem_' tag '.mat']));
    if strcmp(tag,'MSRS'), sd = {A3(1,:,:),A3(2,:,:),A3(3,:,:),A3(4,:,:),A3(5,:,:)};
    else, sd = {P.ACC(strcmp(P.methods,'Ours'),:,:), X.R.(tag)(1,:,:), X.R.(tag)(2,:,:), C.(tag)(1,:,:), C.(tag)(2,:,:)}; end
    for s = 1:5
        for mm = {'SSIM','Qabf','Nabf','VIF','MI','EN'}
            j = find(strcmp(P.names,mm{1})); low = strcmp(mm{1},'Nabf');
            ours = squeeze(sd{s}(1,j,:));
            v6 = cellfun(@(q) mean(P.ACC(strcmp(P.methods,q),j,:)), deep);
            v8 = [v6, cellfun(@(q) mean(P.ACC(strcmp(P.methods,q),j,:)), light)];
            if low, [bv,bq] = min(v6); r6 = 1+sum(v6 < mean(ours)); r8 = 1+sum(v8 < mean(ours));
            else,   [bv,bq] = max(v6); r6 = 1+sum(v6 > mean(ours)); r8 = 1+sum(v8 > mean(ours)); end
            ob = squeeze(P.ACC(strcmp(P.methods,deep{bq}),j,:));
            om = squeeze(P.ACC(strcmp(P.methods,'MMDRFuse'),j,:)); ol = squeeze(P.ACC(strcmp(P.methods,'LUTFuse'),j,:));
            fprintf(fid,'%s,%d,%s,%.6f,%d,%s,%.6f,%.3g,%d,%.6f,%.3g,%.6f,%.3g\n',tag,s-1,mm{1},mean(ours),r6,deep{bq},bv, ...
                signrank(ours,ob),r8,mean(om),signrank(ours,om),mean(ol),signrank(ours,ol));
        end
    end
end
fclose(fid);
% robustness ranks for seeds 0-4
RA = load('results/robust_acc.mat'); RS = load('results/robust_seeds_acc.mat');
fid = fopen('results/seeds5_robust.csv','w');
fprintf(fid,'condition,seed,metric,value,rank6,rank8,p_vs_best6\n');
for c = 1:numel(RA.conds)
    for s = 1:5
        for mm = {'SSIM','Qabf','Nabf','VIF','MI'}
            j = find(strcmp(RA.names,mm{1})); low = strcmp(mm{1},'Nabf');
            if s == 1, ours = squeeze(RA.ACC(c,strcmp(RA.meths,'Ours'),j,:));
            elseif s <= 3, ours = squeeze(RS.ACC(c,s-1,j,:));
            else, ours = squeeze(C.robust(c,s-3,j,:)); end
            v6 = cellfun(@(q) mean(RA.ACC(c,strcmp(RA.meths,q),j,:)), deep);
            v8 = [v6, cellfun(@(q) mean(RA.ACC(c,strcmp(RA.meths,q),j,:)), light)];
            if low, [~,bq] = min(v6); r6 = 1+sum(v6 < mean(ours)); r8 = 1+sum(v8 < mean(ours));
            else,   [~,bq] = max(v6); r6 = 1+sum(v6 > mean(ours)); r8 = 1+sum(v8 > mean(ours)); end
            ob = squeeze(RA.ACC(c,strcmp(RA.meths,deep{bq}),j,:));
            fprintf(fid,'%s,%d,%s,%.6f,%d,%d,%.3g\n',RA.conds{c},s-1,mm{1},mean(ours),r6,r8,signrank(ours,ob));
        end
    end
end
fclose(fid);
type('results/seeds5_metrics.csv'); type('results/seeds5_a3_vs_a1.csv');
end

function [ir,vi,irg,vig] = readpair(dir_ir, dir_vi, nm)
ir = imread(fullfile(dir_ir,nm)); vi = imread(fullfile(dir_vi,nm));
irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
end

function [pr,pp,cr,pc] = classRatio(net, files)
P = []; Cc = [];
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
    if nnz(lab==1) >= 50, Cc(end+1,:) = [mean(W(lab==1)) bg]; end %#ok<AGROW>
end
pr = mean(P(:,1))/mean(P(:,2)); pp = signrank(P(:,1),P(:,2));
cr = mean(Cc(:,1))/mean(Cc(:,2)); pc = signrank(Cc(:,1),Cc(:,2));
end

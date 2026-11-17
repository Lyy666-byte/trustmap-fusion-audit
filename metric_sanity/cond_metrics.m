function cond_metrics(budgetSec, nWorkers)
%COND_METRICS  metric_full on every condition rendered by make_conditions.py, for the images in
%   results/image_sets.json (metric_msrs and roadscene). Same pixels as the segmentation and the human study.
%   Resumable within a time budget; results/cond_metrics.mat, then results/cond_metrics.csv.
if nargin < 1, budgetSec = 6500; end
if nargin < 2, nWorkers = 6; end
t0 = tic;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','ir_vis_fusion'));
S = jsondecode(fileread(fullfile(here,'results','image_sets.json')));
methods = {'DCEvo','EMMA','LUTFuse','MMDRFuse'};
conds = {'native','blend25','blend50','blend75','noise1','noise3','sharp1','sharp2','gamma07','gamma14','contrast','blur1'};
J = cell(0,4);
for d = 1:2
    if d == 1, ds = 'MSRS'; names = S.metric_msrs; else, ds = 'RoadScene'; names = S.roadscene; end
    for i = 1:numel(names)
        for c = {'ir','vi','avg'}, J(end+1,:) = {ds,'base',c{1},names{i}}; end %#ok<AGROW>
        for m = 1:numel(methods)
            for c = 1:numel(conds), J(end+1,:) = {ds,methods{m},conds{c},names{i}}; end %#ok<AGROW>
        end
    end
end
N = size(J,1);
[~,mn] = metric_full(rand(64),rand(64),rand(64));
af = fullfile(here,'results','cond_metrics.mat');
if exist(af,'file'), L = load(af); V = L.V; else, V = nan(N,numel(mn)); end
if isempty(gcp('nocreate')), parpool('local',nWorkers); end
batch = 4*nWorkers;
while any(isnan(V(:,1)))
    if toc(t0) > budgetSec, save(af,'V','J','mn'); fprintf('PARTIAL %d/%d\n',nnz(~isnan(V(:,1))),N); return; end
    todo = find(isnan(V(:,1)),batch)';
    v = nan(numel(todo),numel(mn));
    Jt = J(todo,:);
    parfor k = 1:numel(todo)
        b = fullfile(here,'stim',Jt{k,1},'base');
        ir = im2double(imread(fullfile(b,'ir',Jt{k,4}))); vi = im2double(imread(fullfile(b,'vi',Jt{k,4})));
        F = im2double(imread(fullfile(here,'stim',Jt{k,1},Jt{k,2},Jt{k,3},Jt{k,4})));
        v(k,:) = metric_full(F,ir,vi);
    end
    V(todo,:) = v;
    save(af,'V','J','mn');
end
fid = fopen(fullfile(here,'results','cond_metrics.csv'),'w');
fprintf(fid,'dataset,method,condition,image,%s\n',strjoin(mn,','));
for i = 1:N
    row = sprintf('%.6f,',V(i,:));
    fprintf(fid,'%s,%s,%s,%s,%s\n',J{i,1},J{i,2},J{i,3},J{i,4},row(1:end-1));
end
fclose(fid);
fprintf('FINALIZED cond_metrics\n');
end

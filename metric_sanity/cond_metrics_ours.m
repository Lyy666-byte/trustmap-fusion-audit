function cond_metrics_ours(nWorkers)
%COND_METRICS_OURS  metric_full on the native images of the proposed network (ANALYSIS_PLAN.md, addendum 2026-11-16):
%   the metric sets of MSRS (100), RoadScene (30) and FMB (100). results/cond_metrics_ours.csv, same columns as the others.
if nargin < 1, nWorkers = 6; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','ir_vis_fusion'));
S = jsondecode(fileread(fullfile(here,'results','image_sets.json')));
Sf = jsondecode(fileread(fullfile(here,'results','image_sets_fmb.json')));
J = [repmat({'MSRS'},numel(S.metric_msrs),1) S.metric_msrs(:);
     repmat({'RoadScene'},numel(S.roadscene),1) S.roadscene(:);
     repmat({'FMB'},numel(Sf.metric_fmb),1) Sf.metric_fmb(:)];
N = size(J,1);
[~,mn] = metric_full(rand(64),rand(64),rand(64));
V = nan(N,numel(mn));
if isempty(gcp('nocreate')), parpool('local',nWorkers); end
parfor k = 1:N
    b = fullfile(here,'stim',J{k,1},'base');
    ir = im2double(imread(fullfile(b,'ir',J{k,2}))); vi = im2double(imread(fullfile(b,'vi',J{k,2})));
    F = im2double(imread(fullfile(here,'stim',J{k,1},'Ours','native',J{k,2})));
    V(k,:) = metric_full(F,ir,vi);
end
fid = fopen(fullfile(here,'results','cond_metrics_ours.csv'),'w');
fprintf(fid,'dataset,method,condition,image,%s\n',strjoin(mn,','));
for i = 1:N
    row = sprintf('%.6f,',V(i,:));
    fprintf(fid,'%s,Ours,native,%s,%s\n',J{i,1},J{i,2},row(1:end-1));
end
fclose(fid);
fprintf('FINALIZED cond_metrics_ours %d rows\n',N);
end

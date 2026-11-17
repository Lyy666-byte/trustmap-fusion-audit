function eval_all(irDir,viDir,tag,seaDir,cddDir,c0,c1)
%EVAL_ALL  Fair same-protocol comparison incl. recent SOTA (precomputed folders).
%   File list = intersection of seaDir & cddDir that also exist in irDir/viDir.
%   Metrics are computed at FULL resolution via metric_full.
%   Chunked & resumable: results accumulate in results/peritem_<tag>.mat over
%   several calls (use c0/c1 to pick an index range). When every cell is filled
%   the function finalizes: writes mean/std tables, paired Wilcoxon significance
%   (Ours vs CDDFuse, Ours vs SeAFusion) and per-image curve CSVs.
So = load('results/fusion_net_v3.mat'); ournet = So.net;
Sd = load('results/densefuse.mat');     enc = Sd.enc; dec = Sd.dec;

methods = {'average','laplacian','dwt','gff','DenseFuse','SeAFusion','CDDFuse','Ours'};
names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};

ds = dir(fullfile(seaDir,'*.png')); seaN = {ds.name};
dc = dir(fullfile(cddDir,'*.png')); cddN = {dc.name};
files = intersect(seaN,cddN);
keep = false(size(files));
for i = 1:numel(files)
    keep(i) = (exist(fullfile(irDir,files{i}),'file')>0) && (exist(fullfile(viDir,files{i}),'file')>0);
end
files = sort(files(keep));
N = numel(files);
M = numel(methods); K = numel(names);

pf = fullfile('results',sprintf('peritem_%s.mat',tag));
if exist(pf,'file'), L = load(pf); ACC = L.ACC; else, ACC = nan(M,K,N); end

if nargin < 6 || isempty(c0), c0 = 1; end
if nargin < 7 || isempty(c1), c1 = N; end
c1 = min(c1,N);

for i = c0:c1
    if ~any(isnan(ACC(:,:,i)),'all')   % already computed -> skip (resumable)
        fprintf('skip %d/%d %s\n',i,N,files{i}); continue
    end
    nm = files{i};
    ir = imread(fullfile(irDir,nm)); vi = imread(fullfile(viDir,nm));
    irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
    vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
    for k = 1:M
        switch methods{k}
            case 'Ours',      F = fuse_images(ournet,ir,vi);
            case 'DenseFuse', F = densefuse_fuse(enc,dec,ir,vi);
            case 'SeAFusion', F = imread(fullfile(seaDir,nm));
            case 'CDDFuse',   F = imread(fullfile(cddDir,nm));
            otherwise,        F = classic_fusion(irg,vig,methods{k});
        end
        ACC(k,:,i) = metric_full(F,irg,vig);
    end
    save(pf,'ACC','files','methods','names');   % incremental save (timeout-safe)
    fprintf('done %d/%d %s\n',i,N,nm);
end

if any(isnan(ACC(:)))
    nrem = sum(any(any(isnan(ACC),1),2));
    fprintf('PARTIAL %s saved (%d images still empty)\n',tag,nrem);
    return
end

MU = mean(ACC,3,'omitnan'); SDv = std(ACC,0,3,'omitnan');
T  = array2table(MU, 'VariableNames',names,'RowNames',methods);
TS = array2table(SDv,'VariableNames',names,'RowNames',methods);
writetable(T, fullfile('results',sprintf('metrics_sota_%s.csv',tag)),    'WriteRowNames',true);
writetable(TS,fullfile('results',sprintf('metrics_sota_%s_std.csv',tag)),'WriteRowNames',true);

io = find(strcmp(methods,'Ours'));
ic = find(strcmp(methods,'CDDFuse'));
is = find(strcmp(methods,'SeAFusion'));
fid = fopen(fullfile('results',sprintf('signif_%s.csv',tag)),'w');
fprintf(fid,'metric,Ours_mean,CDD_mean,p_vs_CDD,SeA_mean,p_vs_SeA\n');
for j = 1:K
    o = squeeze(ACC(io,j,:)); c = squeeze(ACC(ic,j,:)); s = squeeze(ACC(is,j,:));
    mc = ~isnan(o) & ~isnan(c); ms = ~isnan(o) & ~isnan(s);
    pc = NaN; ps = NaN;
    try, if sum(mc)>=5, pc = signrank(o(mc),c(mc)); end, catch, end
    try, if sum(ms)>=5, ps = signrank(o(ms),s(ms)); end, catch, end
    fprintf(fid,'%s,%.4f,%.4f,%.4g,%.4f,%.4g\n',names{j}, ...
        mean(o,'omitnan'),mean(c,'omitnan'),pc,mean(s,'omitnan'),ps);
end
fclose(fid);

plotm = {'EN','MI','SSIM','Qabf','SCD','VIF'};
for pm = 1:numel(plotm)
    jj = find(strcmp(names,plotm{pm}));
    fid = fopen(fullfile('results',sprintf('curveA_%s_%s.csv',tag,plotm{pm})),'w');
    fprintf(fid,'%s',methods{1}); fprintf(fid,',%s',methods{2:end}); fprintf(fid,'\n');
    for i = 1:N
        row = squeeze(ACC(:,jj,i));
        fprintf(fid,'%.10g',row(1)); fprintf(fid,',%.10g',row(2:end)); fprintf(fid,'\n');
    end
    fclose(fid);
end
fprintf('FINALIZED %s: tables+signif+curves written (N=%d)\n',tag,N);
end

function refresh_ours(irDir,viDir,tag)
%REFRESH_OURS  Recompute only the 'Ours' row in peritem_<tag>.mat with the
%   current final network (results/fusion_net_v3.mat), reusing every other
%   method's cached metrics, then regenerate tables / significance / curves.
So = load('results/fusion_net_v3.mat'); net = So.net;
pf = fullfile('results',sprintf('peritem_%s.mat',tag));
L = load(pf); ACC = L.ACC; files = L.files; methods = L.methods; names = L.names;
[~,K,N] = size(ACC);

io = find(strcmp(methods,'Ours'));
if isempty(io), error('no Ours row in %s',tag); end

for i = 1:N
    nm = files{i};
    ir = imread(fullfile(irDir,nm)); vi = imread(fullfile(viDir,nm));
    irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
    vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
    F = fuse_images(net,ir,vi);
    ACC(io,:,i) = metric_full(F,irg,vig);
    fprintf('Ours %d/%d %s\n',i,N,nm);
end
save(pf,'ACC','files','methods','names');

MU = mean(ACC,3,'omitnan'); SDv = std(ACC,0,3,'omitnan');
writetable(array2table(MU,'VariableNames',names,'RowNames',methods), ...
    fullfile('results',sprintf('metrics_sota_%s.csv',tag)),'WriteRowNames',true);
writetable(array2table(SDv,'VariableNames',names,'RowNames',methods), ...
    fullfile('results',sprintf('metrics_sota_%s_std.csv',tag)),'WriteRowNames',true);

recent = {'CDDFuse','SeAFusion','EMMA','DCEvo'};
recent = recent(ismember(recent,methods));
fid = fopen(fullfile('results',sprintf('signif_%s.csv',tag)),'w');
fprintf(fid,'metric,Ours_mean');
for r = 1:numel(recent), fprintf(fid,',%s_mean,p_vs_%s',recent{r},recent{r}); end
fprintf(fid,'\n');
for j = 1:K
    o = squeeze(ACC(io,j,:));
    fprintf(fid,'%s,%.4f',names{j},mean(o,'omitnan'));
    for r = 1:numel(recent)
        b = squeeze(ACC(strcmp(methods,recent{r}),j,:));
        fprintf(fid,',%.4f,%.4g',mean(b,'omitnan'),sr(o,b));
    end
    fprintf(fid,'\n');
end
fclose(fid);

plotm = {'EN','MI','SSIM','Qabf','SCD','VIF'};
for pm = 1:numel(plotm)
    jj = find(strcmp(names,plotm{pm}));
    fid = fopen(fullfile('results',sprintf('curveA_%s_%s.csv',tag,plotm{pm})),'w');
    fprintf(fid,'%s',methods{1}); fprintf(fid,',%s',methods{2:end}); fprintf(fid,'\n');
    for i = 1:N
        rr = squeeze(ACC(:,jj,i));
        fprintf(fid,'%.10g',rr(1)); fprintf(fid,',%.10g',rr(2:end)); fprintf(fid,'\n');
    end
    fclose(fid);
end
fprintf('REFRESHED Ours in %s (N=%d)\n',tag,N);
end

function p = sr(a,b)
m = ~isnan(a) & ~isnan(b); p = NaN;
try, if sum(m)>=5, p = signrank(a(m),b(m)); end, catch, end
end

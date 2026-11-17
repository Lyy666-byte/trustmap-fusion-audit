function finalize_tag(tag,ACC,files,methods,names)
%FINALIZE_TAG  Write mean/std tables, paired-Wilcoxon significance and per-image
%   curve CSVs from a per-item metric array. Shared by eval_all / refresh_ours /
%   refresh_mi so every path produces byte-identical outputs.
%
%   The significance file covers EVERY deep method the paper compares against
%   (DenseFuse included), so no claim in the text refers to an untested pair.

[~,K,N] = size(ACC);

MU = mean(ACC,3,'omitnan'); SDv = std(ACC,0,3,'omitnan');
writetable(array2table(MU,'VariableNames',names,'RowNames',methods), ...
    fullfile('results',sprintf('metrics_sota_%s.csv',tag)),'WriteRowNames',true);
writetable(array2table(SDv,'VariableNames',names,'RowNames',methods), ...
    fullfile('results',sprintf('metrics_sota_%s_std.csv',tag)),'WriteRowNames',true);

io = find(strcmp(methods,'Ours'));
deep = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo'};
deep = deep(ismember(deep,methods));
fid = fopen(fullfile('results',sprintf('signif_%s.csv',tag)),'w');
fprintf(fid,'metric,Ours_mean');
for r = 1:numel(deep), fprintf(fid,',%s_mean,p_vs_%s',deep{r},deep{r}); end
fprintf(fid,'\n');
for j = 1:K
    o = squeeze(ACC(io,j,:));
    fprintf(fid,'%s,%.4f',names{j},mean(o,'omitnan'));
    for r = 1:numel(deep)
        b = squeeze(ACC(strcmp(methods,deep{r}),j,:));
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
fprintf('FINALIZED %s (N=%d, %d methods, signif vs %d deep methods)\n', ...
    tag,N,numel(methods),numel(deep));
end

function p = sr(a,b)
m = ~isnan(a) & ~isnan(b); p = NaN;
try, if sum(m)>=5, p = signrank(a(m),b(m)); end, catch, end
end

function finalize_from_cache()
%FINALIZE_FROM_CACHE  Regenerate every CSV the paper reads, from the cached
%   per-image metric arrays, without recomputing any metric. Seconds, not hours.
%
%   Inputs : results/peritem_{MSRS,TNO,RoadScene}.mat, results/abl2_acc.mat
%   Outputs: metrics_sota_<tag>.csv (+_std), signif_<tag>.csv, curveA_<tag>_*.csv,
%            ablation2_MSRS.csv, ablation2_signif.csv, ablation2_trustdev.csv

for tag = {'MSRS','TNO','RoadScene'}
    t = tag{1};
    pf = fullfile('results',sprintf('peritem_%s.mat',t));
    if ~exist(pf,'file'), fprintf('skip %s (no cache)\n',t); continue; end
    L = load(pf);
    finalize_tag(t,L.ACC,L.files,L.methods,L.names);
end

af = 'results/abl2_acc.mat';
if exist(af,'file')
    L = load(af); ACC = L.ACC; DEV = L.DEV; names = L.names; labs = L.labs;
    MU = mean(ACC,3,'omitnan');
    writetable(array2table(MU,'VariableNames',names,'RowNames',labs), ...
        'results/ablation2_MSRS.csv','WriteRowNames',true);
    fid = fopen('results/ablation2_signif.csv','w');
    fprintf(fid,'metric,A0,A1,A2,A3,p_A3vsA0,p_A3vsA1,p_A3vsA2\n');
    for j = 1:numel(names)
        v = squeeze(ACC(:,j,:));
        fprintf(fid,'%s,%.4f,%.4f,%.4f,%.4f,%.4g,%.4g,%.4g\n',names{j}, ...
            mean(v(1,:)),mean(v(2,:)),mean(v(3,:)),mean(v(4,:)), ...
            sr(v(4,:),v(1,:)),sr(v(4,:),v(2,:)),sr(v(4,:),v(3,:)));
    end
    fclose(fid);
    writetable(table(labs(:),mean(DEV,2,'omitnan'), ...
        'VariableNames',{'config','trust_dev'}),'results/ablation2_trustdev.csv');
    fprintf('FINALIZED ablation from cache (N=%d)\n',size(ACC,3));
else
    fprintf('skip ablation (no cache)\n');
end
end

function p = sr(a,b)
a=a(:); b=b(:); m=~isnan(a)&~isnan(b); p=NaN;
try, if sum(m)>=5, p=signrank(a(m),b(m)); end, catch, end
end

function seed_pairs()
%SEED_PAIRS  Per-seed paired comparisons of the full model (A3, seeds 0-4) with every other deep method, from the
%   8-bit caches (no metric is recomputed). MSRS: seedq_acc.mat rows 1-5; TNO and RoadScene: seed 0 from the
%   per-image cache, seeds 1-2 from seeds_xdata_acc.mat, seeds 3-4 from seeds5_acc.mat. Writes results/seed_pairs.csv.
Q = load(fullfile('results','seedq_acc.mat'));
X = load(fullfile('results','seeds_xdata_acc.mat'));
S5 = load(fullfile('results','seeds5_acc.mat')); C = S5.C;
others = {'DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo','MMDRFuse','LUTFuse'};
fid = fopen(fullfile('results','seed_pairs.csv'),'w');
fprintf(fid,'dataset,seed,metric,method,ours,other,p\n');
for tagc = {'MSRS','TNO','RoadScene'}
    tag = tagc{1}; P = load(fullfile('results',['peritem_' tag '.mat']));
    assert(isfield(P,'requant') && P.requant, 'per-image cache of %s is not the 8-bit one', tag);
    if strcmp(tag,'MSRS')
        assert(isequal(Q.files(:),P.files(:)));
        sd = {Q.ACC(1,:,:),Q.ACC(2,:,:),Q.ACC(3,:,:),Q.ACC(4,:,:),Q.ACC(5,:,:)};
        d0 = max(abs(squeeze(Q.ACC(1,:,:)) - squeeze(P.ACC(strcmp(P.methods,'Ours'),:,:))),[],'all');
        assert(d0 < 1e-9, 'seed 0 of seedq differs from the per-image cache (%.2e)', d0);
    else
        sd = {P.ACC(strcmp(P.methods,'Ours'),:,:), X.R.(tag)(1,:,:), X.R.(tag)(2,:,:), C.(tag)(1,:,:), C.(tag)(2,:,:)};
    end
    for s = 1:5
        assert(~any(isnan(sd{s}(:))), 'seed %d of %s incomplete', s-1, tag);
        for j = 1:numel(P.names)
            ours = squeeze(sd{s}(1,j,:));
            for k = 1:numel(others)
                o = squeeze(P.ACC(strcmp(P.methods,others{k}),j,:));
                fprintf(fid,'%s,%d,%s,%s,%.6f,%.6f,%.4g\n',tag,s-1,P.names{j},others{k},mean(ours),mean(o),signrank(ours,o));
            end
        end
    end
end
fclose(fid);
fprintf('DONE seed_pairs\n');
end

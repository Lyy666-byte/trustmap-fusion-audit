function refresh_mi(tag,irDir,viDir,seaDir,cddDir,emmaDir,dcevoDir)
%REFRESH_MI  Recompute ONLY the MI column of a cached peritem_<tag>.mat with the
%   corrected mutual-information estimator (see analysis_MI.m bug history), then
%   rewrite the mean/std tables, the significance file and the per-image curves.
%   The other twelve metrics are expensive and unaffected, so they are reused.
%
%   Fusion of every method is reproduced exactly as eval_all.m produced it, and
%   the pre-processing mirrors metric_full.m so MI is consistent with the rest.

pf = fullfile('results',sprintf('peritem_%s.mat',tag));
L = load(pf); ACC = L.ACC; files = L.files; methods = L.methods; names = L.names;
jMI = find(strcmp(names,'MI'));
if isempty(jMI), error('no MI column'); end
[M,~,N] = size(ACC);

So = load('results/fusion_net_v3.mat'); ournet = So.net;
Sd = load('results/densefuse.mat');     enc = Sd.enc; dec = Sd.dec;
dirmap = struct('SeAFusion',seaDir,'CDDFuse',cddDir,'EMMA',emmaDir,'DCEvo',dcevoDir);

old = squeeze(ACC(:,jMI,:));
for i = 1:N
    nm = files{i};
    ir = imread(fullfile(irDir,nm)); vi = imread(fullfile(viDir,nm));
    for k = 1:M
        switch methods{k}
            case 'Ours',      F = fuse_images(ournet,ir,vi);
            case 'DenseFuse', F = densefuse_fuse(enc,dec,ir,vi);
            case {'SeAFusion','CDDFuse','EMMA','DCEvo'}
                F = imread(fullfile(dirmap.(methods{k}),nm));
            otherwise
                irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
                vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
                F = classic_fusion(irg,vig,methods{k});
        end
        ACC(k,jMI,i) = mi_only(F,ir,vi);
    end
    fprintf('MI %d/%d %s\n',i,N,nm);
end
save(pf,'ACC','files','methods','names');
fprintf('mean MI before %.3f -> after %.3f\n',mean(old(:),'omitnan'),mean(squeeze(ACC(:,jMI,:)),'all','omitnan'));

finalize_tag(tag,ACC,files,methods,names);
end

% -------------------------------------------------------------------------
function v = mi_only(F,ir,vis)
% identical pre-processing to metric_full.m
Fd = im2double(F); ird = im2double(ir); visd = im2double(vis);
if size(Fd,3)>1,   Fd   = rgb2gray(Fd);   end
if size(ird,3)>1,  ird  = rgb2gray(ird);  end
if size(visd,3)>1, visd = rgb2gray(visd); end
if ~isequal(size(Fd),size(ird)),  ird  = imresize(ird, size(Fd)); end
if ~isequal(size(Fd),size(visd)), visd = imresize(visd,size(Fd)); end
v = analysis_MI(im2uint8(ird),im2uint8(visd),im2uint8(Fd));
end

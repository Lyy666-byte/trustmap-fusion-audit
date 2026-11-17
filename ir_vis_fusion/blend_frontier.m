function blend_frontier(budgetSec)
%BLEND_FRONTIER  Metric sanity check: blending a fused image with the pixel average of the sources.
%   For every deep method (ours = reported model, DenseFuse, SeAFusion, CDDFuse, EMMA, DCEvo, MMDRFuse,
%   LUT-Fuse) and a in {0.15, 0.30, 0.50}, F_a = (1-a) F + a (I_ir + I_vis)/2 is scored with SSIM,
%   Qabf, Nabf, MI and VIF (same functions as metric_full). a = 0 is read from the main per-image
%   caches (identical inputs and code). MSRS30 and RoadScene40. Resumable within a time budget;
%   results/blend_acc.mat. When complete, results/blend_frontier.csv (means per method, dataset, a).
if nargin < 1, budgetSec = 515; end
t0 = tic;
as = [0.15 0.30 0.50];
meths = {'Ours','DenseFuse','SeAFusion','CDDFuse','EMMA','DCEvo','MMDRFuse','LUTFuse'};
sets = {'MSRS','tmp_msrs/test/ir','tmp_msrs/test/vi', ...
        {'','','results/sota/SeAFusion_MSRS30','results/sota/CDDFuse_MSRS30','results/sota/EMMA_MSRS', ...
         'results/sota/DCEvo_MSRS','results/sota/MMDRFuse_MSRS','results/sota/LUTFuse_MSRS'}; ...
        'RoadScene','data/RoadScene/ir','data/RoadScene/vi', ...
        {'','','results/sota/SeAFusion_RoadScene','results/sota/CDDFuse_RoadScene','results/sota/EMMA_RoadScene', ...
         'results/sota/DCEvo_RoadScene','results/sota/MMDRFuse_RoadScene','results/sota/LUTFuse_RoadScene'}};
mnames = {'SSIM','Qabf','Nabf','MI','VIF'};
So = load('results/fusion_net_v3.mat'); net = So.net;
Sd = load('results/densefuse.mat'); enc = Sd.enc; dec = Sd.dec;
af = 'results/blend_acc.mat';
if exist(af,'file'), L = load(af); R = L.R; else, R = struct(); end
for d = 1:size(sets,1)
    tag = sets{d,1};
    P = load(fullfile('results',['peritem_' tag '.mat'])); files = P.files; N = numel(files);
    if ~isfield(R,tag), R.(tag) = nan(numel(meths),numel(as),numel(mnames),N); end
    for i = 1:N
        if ~any(isnan(R.(tag)(:,:,:,i)),'all'), continue; end
        if toc(t0) > budgetSec, save(af,'R','as','meths','mnames'); fprintf('PARTIAL (budget)\n'); return; end
        nm = files{i};
        ir = imread(fullfile(sets{d,2},nm)); vi = imread(fullfile(sets{d,3},nm));
        irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
        vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
        avg = (irg + vig)/2;
        for k = 1:numel(meths)
            switch meths{k}
                case 'Ours',      F = fuse_images(net,ir,vi);
                case 'DenseFuse', F = densefuse_fuse(enc,dec,ir,vi);
                otherwise
                    F = im2double(imread(fullfile(sets{d,4}{k},nm)));
                    if size(F,3)>1, F = rgb2gray(F); end
            end
            for q = 1:numel(as)
                Fa = im2double(im2uint8((1-as(q))*F + as(q)*avg));   % scored as an 8-bit image, as in metric_full
                Fu = im2uint8(Fa); iru = im2uint8(irg); viu = im2uint8(vig);
                R.(tag)(k,q,:,i) = [(ssim(Fa,irg)+ssim(Fa,vig))/2, analysis_Qabf(irg,vig,Fa), ...
                    analysis_nabf(Fa,irg,vig), analysis_MI(iru,viu,Fu), vifp(irg*255,Fa*255)+vifp(vig*255,Fa*255)];
            end
        end
        save(af,'R','as','meths','mnames');
        fprintf('%s %2d/%d %s\n', tag, i, N, nm);
    end
end
fid = fopen('results/blend_frontier.csv','w');
fprintf(fid,'dataset,method,a,%s\n',strjoin(mnames,','));
for d = 1:size(sets,1)
    tag = sets{d,1};
    P = load(fullfile('results',['peritem_' tag '.mat']));
    for k = 1:numel(meths)
        a0 = cellfun(@(m) mean(P.ACC(strcmp(P.methods,meths{k}),strcmp(P.names,m),:)), mnames);
        fprintf(fid,'%s,%s,0.00,%s\n',tag,meths{k},strjoin(arrayfun(@(x) sprintf('%.6f',x),a0,'uni',0),','));
        for q = 1:numel(as)
            v = squeeze(mean(R.(tag)(k,q,:,:),4))';
            fprintf(fid,'%s,%s,%.2f,%s\n',tag,meths{k},as(q),strjoin(arrayfun(@(x) sprintf('%.6f',x),v,'uni',0),','));
        end
    end
end
fclose(fid);
type('results/blend_frontier.csv');
fprintf('FINALIZED blend_frontier\n');
end

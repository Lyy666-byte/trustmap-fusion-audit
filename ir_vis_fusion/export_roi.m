function export_roi(irDir,viDir,seaDir,cddDir,emmaDir,dcevoDir,name,tag)
%EXPORT_ROI  Export the 9 panels (IR,VIS,GFF,DenseFuse,SeAFusion,CDDFuse,EMMA,
%   DCEvo,Ours) at full resolution for one scene, with Ours produced by the
%   current final model (results/fusion_net_v3.mat = V7). Real outputs only.
So = load('results/fusion_net_v3.mat'); ournet = So.net;
Sd = load('results/densefuse.mat');     enc = Sd.enc; dec = Sd.dec;
outdir = 'results/export2'; if ~exist(outdir,'dir'), mkdir(outdir); end

ir = imread(fullfile(irDir,name)); vi = imread(fullfile(viDir,name));
irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
imwrite(irg, fullfile(outdir,[tag '_ir.png']));
imwrite(vig, fullfile(outdir,[tag '_vis.png']));
imwrite(classic_fusion(irg,vig,'gff'),  fullfile(outdir,[tag '_gff.png']));
imwrite(densefuse_fuse(enc,dec,ir,vi),  fullfile(outdir,[tag '_densefuse.png']));
rd = @(D) g2(im2double(imread(fullfile(D,name))));
imwrite(rd(seaDir),   fullfile(outdir,[tag '_seafusion.png']));
imwrite(rd(cddDir),   fullfile(outdir,[tag '_cddfuse.png']));
imwrite(rd(emmaDir),  fullfile(outdir,[tag '_emma.png']));
imwrite(rd(dcevoDir), fullfile(outdir,[tag '_dcevo.png']));
imwrite(fuse_images(ournet,ir,vi), fullfile(outdir,[tag '_ours.png']));
fprintf('export_roi done %s (%s)\n',tag,name);
end
function y = g2(x), if size(x,3)>1, y = rgb2gray(x); else, y = x; end, end

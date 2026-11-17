function export2(irDir,viDir,seaDir,cddDir,name,tag)
%EXPORT2  Save full-res fused images of all methods (incl. SOTA) for one scene.
So = load('results/fusion_net_v3.mat'); ournet = So.net;
Sd = load('results/densefuse.mat');     enc = Sd.enc; dec = Sd.dec;
outdir = 'results/export2'; if ~exist(outdir,'dir'), mkdir(outdir); end

ir = imread(fullfile(irDir,name)); vi = imread(fullfile(viDir,name));
irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
imwrite(irg, fullfile(outdir,[tag '_ir.png']));
imwrite(vig, fullfile(outdir,[tag '_vis.png']));
imwrite(classic_fusion(irg,vig,'average'),   fullfile(outdir,[tag '_average.png']));
imwrite(classic_fusion(irg,vig,'laplacian'), fullfile(outdir,[tag '_laplacian.png']));
imwrite(classic_fusion(irg,vig,'dwt'),       fullfile(outdir,[tag '_dwt.png']));
imwrite(classic_fusion(irg,vig,'gff'),       fullfile(outdir,[tag '_gff.png']));
imwrite(densefuse_fuse(enc,dec,ir,vi),       fullfile(outdir,[tag '_densefuse.png']));
sea = im2double(imread(fullfile(seaDir,name))); if size(sea,3)>1, sea = rgb2gray(sea); end
imwrite(sea, fullfile(outdir,[tag '_seafusion.png']));
imwrite(im2double(imread(fullfile(cddDir,name))), fullfile(outdir,[tag '_cddfuse.png']));
imwrite(fuse_images(ournet,ir,vi),           fullfile(outdir,[tag '_ours.png']));
fprintf('export2 done for %s (%s)\n',tag,name);
end

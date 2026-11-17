function export_maps(nameList)
%EXPORT_MAPS  Export, for several MSRS scenes, the real network outputs needed
%   for the interpretability figures (reviewer comments 4 & 5):
%     * IR, VIS, fused
%     * trust map W (new model, results/fusion_net_v3.mat) -> spatial selection
%     * old CBAM spatial attention (results/fusion_net_v4_OLD_backup.mat)
%     * saliency weight S (loss-side, from IR local contrast) and the IR local
%       contrast itself, to show the S<->contrast correspondence.
%   Everything is a genuine model/measurement output; nothing is hand-drawn.
if nargin < 1 || isempty(nameList)
    nameList = {'00537D.png','00633D.png','01023N.png','00147D.png'};
end
Sn = load('results/fusion_net_v3.mat');        netNew = Sn.net;
So = load('results/fusion_net_v4_OLD_backup.mat'); netOld = So.net;
outdir = 'results/maps'; if ~exist(outdir,'dir'), mkdir(outdir); end

for q = 1:numel(nameList)
    nm = nameList{q};
    irp = fullfile('tmp_msrs/test/ir',nm); vip = fullfile('tmp_msrs/test/vi',nm);
    if ~isfile(irp), fprintf('missing %s, skip\n',nm); continue; end
    ir = im2single(imread(irp)); vi = im2single(imread(vip));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end

    W  = gv(predict(netNew,X,'Outputs','trust'));   % new trust map (raw, no rescale)
    F  = gv(predict(netNew,X,'Outputs','fused'));
    A  = gv(predict(netOld,X,'Outputs','sa_sig'));   % old CBAM spatial attention

    % saliency weight S exactly as used in the loss (IR local contrast, 15x15)
    mu = imfilter(double(ir),ones(15)/225,'replicate');
    sal = abs(double(ir)-mu);
    S = (sal-min(sal(:)))/(max(sal(:))-min(sal(:))+1e-6);
    Cst = sal;    % IR local contrast (unnormalised magnitude) for correspondence

    base = fullfile(outdir,erase(nm,'.png'));
    imwrite(ir,        [base '_ir.png']);
    imwrite(vi,        [base '_vi.png']);
    imwrite(mat2gray(F),[base '_fused.png']);
    imwrite(W,         [base '_trust_raw.png']);        % [0,1] already (sigmoid)
    imwrite(A,         [base '_cbam_raw.png']);         % [0,1] already (sigmoid)
    imwrite(S,         [base '_saliency.png']);
    imwrite(mat2gray(Cst),[base '_ircontrast.png']);
    % also dump the raw arrays for exact stats in the python figure
    save([base '_maps.mat'],'W','A','S','Cst','F','ir','vi');
    fprintf('exported %s : W[min %.3f max %.3f std %.3f]  CBAM[min %.3f max %.3f std %.3f]\n', ...
        nm, min(W(:)),max(W(:)),std(W(:)), min(A(:)),max(A(:)),std(A(:)));
end
end

function y = gv(x), y = double(gather(extractdata(x))); end

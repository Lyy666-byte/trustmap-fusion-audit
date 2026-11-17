function test_constant_map_predecessor(nImg)
%TEST_CONSTANT_MAP_PREDECESSOR  The constant-map intervention of test_constant_w.m,
%   applied to the single-stream predecessor (results/fusion_net_v4_OLD_backup.mat),
%   so that the two networks are tested by the SAME procedure.
%
%   For each of the 30 MSRS test images we read the recalibrated feature tensor
%   ('ca_mul') and the spatial attention map ('sa_sig'), replace the map by
%   (a) its own spatial mean and (b) the constant 1 (no spatial attention), re-run
%   the remaining decoder layers with the trained weights, and evaluate the same
%   unsupervised loss (Eq. (2), lambda_g = 20, lambda_s = 3, lambda_tv = 0.5).
%   Also counts the learnable parameters of the ablation networks A0-A3.
if nargin < 1 || isempty(nImg), nImg = 30; end
S = load('results/fusion_net_v4_OLD_backup.mat'); net = S.net;
L0 = load('results/peritem_MSRS.mat'); files = L0.files;
files = files(1:min(nImg,numel(files)));
N = numel(files);

dec = decoder_subnet(net);          % dec1 ... fused, weights copied

res = zeros(N,3);                   % [real map, map -> its mean, map -> 1]
dev = zeros(N,1);
for i = 1:N
    nm = files{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end

    Fc = predict(net,X,'Outputs','ca_mul');     % recalibrated features (32 ch)
    M  = predict(net,X,'Outputs','sa_sig');     % spatial attention map (1 ch)
    mbar = double(gather(extractdata(mean(M,'all'))));
    Md = double(gather(extractdata(M)));
    dev(i) = norm(Md(:)-mbar)/(norm(Md(:))+1e-9);

    variants = {M, M*0 + mbar, M*0 + 1};
    for v = 1:3
        F = predict(dec, Fc.*variants{v});
        res(i,v) = unsup_loss(F, X(:,:,1,:), X(:,:,2,:));
    end
    % consistency check: the rebuilt decoder must reproduce the network output
    if i == 1
        Fnet = predict(net,X);
        Fdec = predict(dec, Fc.*M);
        fprintf('decoder consistency: max |net - rebuilt| = %.2e\n', ...
            double(gather(extractdata(max(abs(Fnet-Fdec),[],'all')))));
    end
    fprintf('%2d/%2d %s  real %.4f | mean-const %.4f | M=1 %.4f | dev %.4f\n', ...
        i,N,nm,res(i,1),res(i,2),res(i,3),dev(i));
end

mu = mean(res,1);
fprintf('\n===== constant-map test, single-stream predecessor, %d MSRS images =====\n',N);
fprintf('  loss with the real map        : %.4f\n',mu(1));
fprintf('  map replaced by its own mean  : %.4f   (%+.2f%%)\n',mu(2),100*(mu(2)-mu(1))/mu(1));
fprintf('  map forced to 1               : %.4f   (%+.2f%%)\n',mu(3),100*(mu(3)-mu(1))/mu(1));
p = NaN; try, p = signrank(res(:,2),res(:,1)); catch, end
fprintf('  paired signrank (mean-const vs real): p = %.3g\n',p);
fprintf('  images where flattening increased the loss: %d/%d\n',sum(res(:,2)>res(:,1)),N);
fprintf('  mean relative deviation of the map from its mean: %.4f\n',mean(dev));

writetable(array2table([res dev],'VariableNames',{'real','mean_const','M1','dev_from_const'}, ...
    'RowNames',files(:)),'results/constant_map_test_predecessor.csv','WriteRowNames',true);
fprintf('wrote results/constant_map_test_predecessor.csv\n');

% ---- learnable-parameter counts of the ablation configurations -------------
mods = {'results/model_AB0.mat','results/model_AB1.mat','results/model_AB2.mat', ...
        'results/fusion_net_v3.mat','results/fusion_net_v4_OLD_backup.mat'};
labs = {'A0_single','A1_dual_cat','A2_cat_trust','A3_diff_trust','predecessor_CBAM_style'};
np = zeros(numel(mods),1);
for k = 1:numel(mods)
    T = load(mods{k}); f = fieldnames(T); n = T.(f{strcmp(f,'net')});
    np(k) = sum(cellfun(@numel, n.Learnables.Value));
    fprintf('  %-24s %d parameters\n',labs{k},np(k));
end
writetable(table(labs(:),np,'VariableNames',{'config','params'}),'results/ablation2_params.csv');
fprintf('wrote results/ablation2_params.csv\n');
end

% ------------------------------------------------------------------ helpers
function dnet = decoder_subnet(net)
names = {'dec1','dec_r','dec2','fused'};
lg = layerGraph();
inp = functionLayer(@(x) x,'Name','fin','Formattable',false,'Acceleratable',true);
lg = addLayers(lg,inp);
prev = 'fin';
for k = 1:numel(names)
    l = net.Layers(strcmp({net.Layers.Name},names{k}));
    lg = addLayers(lg,l);
    lg = connectLayers(lg,prev,names{k});
    prev = names{k};
end
dnet = dlnetwork(lg,'Initialize',false);
dnet = initialize(dnet, dlarray(zeros(32,32,32,1,'single'),'SSCB'));
Ls = net.Learnables; Ld = dnet.Learnables;
for i = 1:height(Ld)
    m = strcmp(string(Ls.Layer),string(Ld.Layer(i))) & ...
        strcmp(string(Ls.Parameter),string(Ld.Parameter(i)));
    assert(any(m),'weight not found');
    Ld.Value{i} = Ls.Value{find(m,1)};
end
dnet.Learnables = Ld;
end

function L = unsup_loss(F,ir,vis)
%UNSUP_LOSS  Eq. (2) with lambda_g = 20, lambda_s = 3, lambda_tv = 0.5 (as in test_constant_w.m).
S = sal_w(ir);
Lint = mean( S.*abs(F-ir) + (1-S).*abs(F-vis), 'all');
gF = sob(F); gI = sob(ir); gV = sob(vis);
Lgrad = mean(abs(gF - max(gI,gV)),'all');
Lssim = 1 - 0.5*(ssim_t(F,ir) + ssim_t(F,vis));
Ltv = tv(F);
L = double(gather(extractdata(Lint + 20*Lgrad + 3*Lssim + 0.5*Ltv)));
end

function S = sal_w(ir)
w = dlarray(reshape(ones(15,15,'single')/225,[15 15 1 1]));
mu = dlconv(ir,w,0,'Padding','same');
s = abs(ir-mu); mn = min(s,[],[1 2]); mx = max(s,[],[1 2]);
S = (s-mn)./(mx-mn+1e-6);
end
function g = sob(x)
wx = dlarray(reshape(single([-1 0 1;-2 0 2;-1 0 1]),[3 3 1 1]));
wy = dlarray(reshape(single([-1 -2 -1;0 0 0;1 2 1]),[3 3 1 1]));
g = sqrt(dlconv(x,wx,0,'Padding','same').^2 + dlconv(x,wy,0,'Padding','same').^2 + 1e-6);
end
function s = ssim_t(x,y)
C1=0.01^2; C2=0.03^2;
w = dlarray(reshape(ones(7,7,'single')/49,[7 7 1 1]));
mx=dlconv(x,w,0,'Padding','same'); my=dlconv(y,w,0,'Padding','same');
sx=dlconv(x.^2,w,0,'Padding','same')-mx.^2; sy=dlconv(y.^2,w,0,'Padding','same')-my.^2;
sxy=dlconv(x.*y,w,0,'Padding','same')-mx.*my;
s = mean(((2*mx.*my+C1).*(2*sxy+C2))./((mx.^2+my.^2+C1).*(sx+sy+C2)),'all');
end
function t = tv(x)
dx = dlarray(reshape(single([-1 1]),[1 2 1 1]));
dy = dlarray(reshape(single([-1;1]),[2 1 1 1]));
t = mean(abs(dlconv(x,dx,0,'Padding','same'))+abs(dlconv(x,dy,0,'Padding','same')),'all');
end

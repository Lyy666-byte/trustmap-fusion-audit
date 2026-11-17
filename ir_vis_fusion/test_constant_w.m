function res = test_constant_w(nImg, modelFile, outCsv)
%TEST_CONSTANT_W  Empirical test of the paper's central structural claim:
%   that a CONSTANT trust map cannot satisfy the spatially varying
%   saliency-guided objective, so degeneracy is penalised by the loss itself.
%
%   For each test image we take the trained network, read the real trust map W,
%   and re-run the decoder path with W replaced by (a) its own spatial mean and
%   (b) the extreme constants 0 and 1. The unsupervised loss of Eq. (2) is
%   evaluated for each variant. If the argument in Section 3.3 is right, the
%   loss must rise when W is flattened.
%   Optional: modelFile (default the deployed model) and outCsv (default
%   results/constant_w_test.csv), used for the additional training seeds.
if nargin < 1 || isempty(nImg), nImg = 30; end
if nargin < 2 || isempty(modelFile), modelFile = 'results/fusion_net_v3.mat'; end
if nargin < 3 || isempty(outCsv), outCsv = 'results/constant_w_test.csv'; end
S = load(modelFile); net = S.net;
L0 = load('results/peritem_MSRS.mat'); files = L0.files;
files = files(1:min(nImg,numel(files)));
N = numel(files);

dec = decoder_subnet(net);          % rf_c1 ... fused, weights copied

res = zeros(N,4);                   % [real, mean-constant, W=0, W=1]
for i = 1:N
    nm = files{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end

    W    = predict(net,X,'Outputs','trust');
    Dm   = predict(net,X,'Outputs','diffm');
    Fvis = predict(net,X,'Outputs','fvis');
    wbar = double(gather(extractdata(mean(W,'all'))));

    % build the constants from W itself so the dimension labels are preserved
    variants = {W, W*0 + wbar, W*0, W*0 + 1};
    for v = 1:4
        fusef = Fvis + variants{v}.*Dm;
        F = predict(dec, fusef);
        res(i,v) = unsup_loss(F, X(:,:,1,:), X(:,:,2,:));
    end
    fprintf('%2d/%2d %s  real %.4f | mean-const %.4f | W=0 %.4f | W=1 %.4f\n', ...
        i,N,nm,res(i,1),res(i,2),res(i,3),res(i,4));
end

mu = mean(res,1);
fprintf('\n===== constant-trust-map test over %d MSRS images =====\n',N);
fprintf('  loss with the real W          : %.4f\n',mu(1));
fprintf('  W replaced by its own mean    : %.4f   (+%.1f%%)\n',mu(2),100*(mu(2)-mu(1))/mu(1));
fprintf('  W forced to 0 (visible only)  : %.4f   (+%.1f%%)\n',mu(3),100*(mu(3)-mu(1))/mu(1));
fprintf('  W forced to 1 (infrared only) : %.4f   (+%.1f%%)\n',mu(4),100*(mu(4)-mu(1))/mu(1));
p = NaN; try, p = signrank(res(:,2),res(:,1)); catch, end
fprintf('  paired signrank (mean-const vs real): p = %.3g\n',p);
fprintf('  images where flattening W increased the loss: %d/%d\n',sum(res(:,2)>res(:,1)),N);

writetable(array2table(res,'VariableNames',{'real','mean_const','W0','W1'}, ...
    'RowNames',files(:)),outCsv,'WriteRowNames',true);
fprintf('wrote %s\n',outCsv);
end

% ------------------------------------------------------------------ helpers
function dnet = decoder_subnet(net)
%DECODER_SUBNET  Rebuild rf_c1..fused as a standalone network with the trained
%   weights, so the fused feature can be supplied directly.
names = {'rf_c1','rf_r1','rf_c2','rf_add','rf_o','dec1','dec_r','dec2','fused'};
lg = layerGraph();
for k = 1:numel(names)
    l = net.Layers(strcmp({net.Layers.Name},names{k}));
    lg = addLayers(lg,l);
end
lg = connectLayers(lg,'rf_c1','rf_r1');
lg = connectLayers(lg,'rf_r1','rf_c2');
lg = connectLayers(lg,'rf_c2','rf_add/in2');
lg = connectLayers(lg,'rf_add','rf_o');
lg = connectLayers(lg,'rf_o','dec1');
lg = connectLayers(lg,'dec1','dec_r');
lg = connectLayers(lg,'dec_r','dec2');
lg = connectLayers(lg,'dec2','fused');
% the residual skip enters rf_add/in1 straight from the input; a pass-through
% functionLayer is the entry point so the net stays fully convolutional
% (no imageInputLayer => arbitrary image size, as in build_fusion_net_v7.m)
inp = functionLayer(@(x) x,'Name','fin','Formattable',false,'Acceleratable',true);
lg = addLayers(lg,inp);
lg = connectLayers(lg,'fin','rf_c1');
lg = connectLayers(lg,'fin','rf_add/in1');
dnet = dlnetwork(lg,'Initialize',false);
dnet = initialize(dnet, dlarray(zeros(32,32,20,1,'single'),'SSCB'));
% copy trained learnables by (layer,parameter) name
Ls = net.Learnables; Ld = dnet.Learnables;
for i = 1:height(Ld)
    m = strcmp(string(Ls.Layer),string(Ld.Layer(i))) & ...
        strcmp(string(Ls.Parameter),string(Ld.Parameter(i)));
    if any(m), Ld.Value{i} = Ls.Value{find(m,1)}; end
end
dnet.Learnables = Ld;
end

function L = unsup_loss(F,ir,vis)
%UNSUP_LOSS  Eq. (2) with the deployed weights lambda_g=20, ws=3, wtv=0.5.
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

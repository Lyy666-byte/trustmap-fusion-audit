function train_resume(tag,builderName,lossName,totalEpochs,nPairs,maxEpochsThisCall,builderArg,seed)
%TRAIN_RESUME  Resumable Adam training so long runs can be split across calls.
%   Checkpoints net + Adam moments + epoch counter to results/ck_<tag>.mat.
%   Call repeatedly with the same arguments until it prints DONE.
%
%   train_resume('V6','build_fusion_net_v6','fusion_loss_v5',40,200,12,20)
%
%   seed (optional, default 0) sets the initialisation and the mini-batch order. With the
%   default the behaviour is exactly that of the original runs (rng(0) at initialisation only);
%   with seed > 0 the stream is also re-seeded at every resume (rng(1000*seed + epoch)), so a
%   seeded run does not depend on how it is split across calls. The training patches are the
%   same for every seed (results/data_<nPairs>pairs.mat).

if nargin < 4 || isempty(totalEpochs),       totalEpochs = 40;  end
if nargin < 5 || isempty(nPairs),            nPairs      = 200; end
if nargin < 6 || isempty(maxEpochsThisCall), maxEpochsThisCall = 10; end
if nargin < 7, builderArg = []; end
if nargin < 8 || isempty(seed), seed = 0; end

if ~exist('results','dir'), mkdir results; end
ckf = fullfile('results',sprintf('ck_%s.mat',tag));

% ---- data (cached so every chunk sees exactly the same training set) ----
dcf = fullfile('results',sprintf('data_%dpairs.mat',nPairs));
if exist(dcf,'file')
    L = load(dcf); allX = L.allX;
    fprintf('loaded cached patches: %d\n',size(allX,4));
else
    rng(0);
    allX = prepare_data('tmp_msrs/train/ir','tmp_msrs/train/vi',128,128,nPairs);
    save(dcf,'allX','-v7.3');
end

lossFcn = str2func(lossName);
N = size(allX,4);
miniBatchSize = 16; learnRate = 1e-3;

if exist(ckf,'file')
    S = load(ckf);
    net = S.net; avgG = S.avgG; avgSq = S.avgSq; iter = S.iter; e0 = S.epoch;
    fprintf('resume %s from epoch %d/%d\n',tag,e0,totalEpochs);
    if seed > 0, rng(1000*seed + e0); end
else
    rng(seed);
    if isempty(builderArg), net = feval(builderName);
    else,                   net = feval(builderName,builderArg); end
    avgG = []; avgSq = []; iter = 0; e0 = 0;
    np = 0; L2 = net.Learnables;
    for i = 1:height(L2), np = np + numel(L2.Value{i}); end
    fprintf('start %s : %d params\n',tag,np);
end

eEnd = min(totalEpochs, e0 + maxEpochsThisCall);
for epoch = e0+1:eEnd
    idx = randperm(N);
    epochLoss = 0; nb = 0; parts = [0 0 0];
    for s = 1:miniBatchSize:N
        iter = iter + 1; nb = nb + 1;
        b  = idx(s:min(s+miniBatchSize-1,N));
        Xb = dlarray(single(allX(:,:,:,b)),'SSCB');
        if canUseGPU, Xb = gpuArray(Xb); end
        [loss,grads,parts] = dlfeval(lossFcn,net,Xb);
        [net,avgG,avgSq]   = adamupdate(net,grads,avgG,avgSq,iter,learnRate);
        epochLoss = epochLoss + double(gather(extractdata(loss)));
    end
    fprintf('epoch %2d/%2d  loss=%.4f  Lint=%.4f  Lgrad=%.4f\n', ...
            epoch,totalEpochs,epochLoss/nb,parts(1),parts(2));
    save(ckf,'net','avgG','avgSq','iter','epoch');
end

if eEnd >= totalEpochs
    save(fullfile('results',sprintf('model_%s.mat',tag)),'net');
    fprintf('DONE %s -> results/model_%s.mat\n',tag,tag);
else
    fprintf('PARTIAL %s : %d/%d epochs, call again\n',tag,eEnd,totalEpochs);
end
end

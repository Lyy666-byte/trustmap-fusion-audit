function net = train_fusion(net,allX,numEpochs,miniBatchSize,learnRate,lossFcn)
%TRAIN_FUSION  Custom Adam training loop for the unsupervised fusion network.
%   net     : dlnetwork from build_fusion_net()/build_fusion_net_v2()
%   allX    : H x W x 2 x N single array, ch1 = IR, ch2 = VIS, [0,1]
%   lossFcn : (optional) model-loss handle, default @fusion_loss
%
%   Returns the trained dlnetwork.

if nargin < 3 || isempty(numEpochs),     numEpochs     = 30;            end
if nargin < 4 || isempty(miniBatchSize), miniBatchSize = 16;            end
if nargin < 5 || isempty(learnRate),     learnRate     = 1e-3;          end
if nargin < 6 || isempty(lossFcn),       lossFcn       = @fusion_loss;  end

N = size(allX,4);
avgG = []; avgSq = []; iter = 0;
useGPU = canUseGPU;

for epoch = 1:numEpochs
    idx = randperm(N);
    epochLoss = 0; nb = 0;
    for s = 1:miniBatchSize:N
        iter = iter + 1; nb = nb + 1;
        b  = idx(s:min(s+miniBatchSize-1,N));
        Xb = dlarray(single(allX(:,:,:,b)),'SSCB');
        if useGPU, Xb = gpuArray(Xb); end
        [loss,grads,parts] = dlfeval(lossFcn,net,Xb);
        [net,avgG,avgSq]   = adamupdate(net,grads,avgG,avgSq,iter,learnRate);
        epochLoss = epochLoss + double(gather(extractdata(loss)));
    end
    fprintf('epoch %2d/%2d  loss=%.4f  Lint=%.4f  Lgrad=%.4f\n', ...
            epoch,numEpochs,epochLoss/nb,parts(1),parts(2));
end
end

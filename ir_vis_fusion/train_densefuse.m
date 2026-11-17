function [enc,dec] = train_densefuse(enc,dec,allX1,nEpochs,bs,lr)
%TRAIN_DENSEFUSE  Train DenseFuse autoencoder on single-channel patches.
%   allX1 : H x W x 1 x N single, [0,1] (pool IR and VIS patches together).
if nargin < 4 || isempty(nEpochs), nEpochs = 30;   end
if nargin < 5 || isempty(bs),      bs      = 16;   end
if nargin < 6 || isempty(lr),      lr      = 1e-3; end

aE=[]; aSE=[]; aD=[]; aSD=[]; it=0;
useGPU = canUseGPU;
N = size(allX1,4);
for e = 1:nEpochs
    idx = randperm(N);
    last = 0;
    for s = 1:bs:N
        it = it + 1;
        b  = idx(s:min(s+bs-1,N));
        Xb = dlarray(single(allX1(:,:,:,b)),'SSCB');
        if useGPU, Xb = gpuArray(Xb); end
        [loss,gE,gD] = dlfeval(@densefuse_loss,enc,dec,Xb);
        [enc,aE,aSE] = adamupdate(enc,gE,aE,aSE,it,lr);
        [dec,aD,aSD] = adamupdate(dec,gD,aD,aSD,it,lr);
        last = double(gather(extractdata(loss)));
    end
    fprintf('densefuse epoch %2d/%2d  loss=%.5f\n',e,nEpochs,last);
end
end

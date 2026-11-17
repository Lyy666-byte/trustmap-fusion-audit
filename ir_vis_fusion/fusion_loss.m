function [loss,grads,parts] = fusion_loss(net,X)
%FUSION_LOSS  Unsupervised loss for IR-VIS fusion (use inside dlfeval).
%   X : dlarray 'SSCB', 2 channels (1=IR, 2=VIS), values in [0,1].
%
%   loss = L_intensity + lambda * L_gradient
%     L_intensity : keep the brighter (thermal) content  -> |F - max(IR,VIS)|
%     L_gradient  : keep the richer texture/edges         -> |gM(F) - max(gM(IR),gM(VIS))|
%   Both terms are fully differentiable w.r.t. the network weights.

lambda = 10;

F   = forward(net,X);
ir  = X(:,:,1,:);
vis = X(:,:,2,:);

Lint  = mean(abs(F - max(ir,vis)),'all');

gF = sobel_mag(F);
gI = sobel_mag(ir);
gV = sobel_mag(vis);
Lgrad = mean(abs(gF - max(gI,gV)),'all');

loss  = Lint + lambda*Lgrad;
grads = dlgradient(loss,net.Learnables);
parts = [double(extractdata(Lint)), double(extractdata(Lgrad))];
end

function g = sobel_mag(x)
%SOBEL_MAG  Differentiable Sobel gradient magnitude via dlconv (fixed kernels).
wx = dlarray(reshape(single([-1 0 1; -2 0 2; -1 0 1]),[3 3 1 1]));
wy = dlarray(reshape(single([-1 -2 -1; 0 0 0; 1 2 1]),[3 3 1 1]));
gx = dlconv(x,wx,0,'Padding','same');
gy = dlconv(x,wy,0,'Padding','same');
g  = sqrt(gx.^2 + gy.^2 + 1e-6);
end

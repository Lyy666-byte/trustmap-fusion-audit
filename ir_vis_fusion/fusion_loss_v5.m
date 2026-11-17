function [loss,grads,parts] = fusion_loss_v5(net,X)
%FUSION_LOSS_V5  v3 loss + total-variation artifact-suppression term.
%   loss = L_int(saliency) + 20*L_grad + 3*(1-SSIM) + 0.5*L_tv
%   The TV term penalizes spurious gradients (fusion artifacts) and lowers Nabf
%   while the gradient term still preserves genuine source edges.

wg = 20; ws = 3; wtv = 0.5;

F   = forward(net,X);
ir  = X(:,:,1,:);
vis = X(:,:,2,:);

S    = saliency_weight(ir);
Lint = mean( S.*abs(F-ir) + (1-S).*abs(F-vis), 'all');

gF = sobel_mag(F); gI = sobel_mag(ir); gV = sobel_mag(vis);
Lgrad = mean(abs(gF - max(gI,gV)),'all');

Lssim = 1 - 0.5*(ssim_term(F,ir) + ssim_term(F,vis));

Ltv = tv_term(F);

loss  = Lint + wg*Lgrad + ws*Lssim + wtv*Ltv;
grads = dlgradient(loss,net.Learnables);
parts = [double(extractdata(Lint)), double(extractdata(Lgrad)), double(extractdata(Ltv))];
end

% -------------------------------------------------------------------------
function S = saliency_weight(ir)
w  = dlarray(reshape(ones(15,15,'single')/225,[15 15 1 1]));
mu = dlconv(ir,w,0,'Padding','same');
sal = abs(ir - mu);
mn = min(sal,[],[1 2]); mx = max(sal,[],[1 2]);
S  = (sal - mn) ./ (mx - mn + 1e-6);
end

function g = sobel_mag(x)
wx = dlarray(reshape(single([-1 0 1; -2 0 2; -1 0 1]),[3 3 1 1]));
wy = dlarray(reshape(single([-1 -2 -1; 0 0 0; 1 2 1]),[3 3 1 1]));
gx = dlconv(x,wx,0,'Padding','same');
gy = dlconv(x,wy,0,'Padding','same');
g  = sqrt(gx.^2 + gy.^2 + 1e-6);
end

function s = ssim_term(x,y)
C1 = 0.01^2; C2 = 0.03^2;
w   = dlarray(reshape(ones(7,7,'single')/49,[7 7 1 1]));
mux = dlconv(x,w,0,'Padding','same');
muy = dlconv(y,w,0,'Padding','same');
sx  = dlconv(x.^2,w,0,'Padding','same') - mux.^2;
sy  = dlconv(y.^2,w,0,'Padding','same') - muy.^2;
sxy = dlconv(x.*y,w,0,'Padding','same') - mux.*muy;
ssim_map = ((2*mux.*muy + C1).*(2*sxy + C2)) ./ ...
           ((mux.^2 + muy.^2 + C1).*(sx + sy + C2));
s = mean(ssim_map,'all');
end

function t = tv_term(x)
dxk = dlarray(reshape(single([-1 1]),[1 2 1 1]));
dyk = dlarray(reshape(single([-1;1]),[2 1 1 1]));
gx = dlconv(x,dxk,0,'Padding','same');
gy = dlconv(x,dyk,0,'Padding','same');
t  = mean(abs(gx) + abs(gy),'all');
end

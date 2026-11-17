function [loss,gE,gD] = densefuse_loss(enc,dec,X)
%DENSEFUSE_LOSS  Autoencoder reconstruction loss (pixel + SSIM).
feat = forward(enc,X);
rec  = forward(dec,feat);
Lpix  = mean((rec - X).^2,'all');
Lssim = 1 - ssim_term(rec,X);
loss  = Lpix + 0.1*Lssim;
[gE,gD] = dlgradient(loss, enc.Learnables, dec.Learnables);
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

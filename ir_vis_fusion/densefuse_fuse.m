function F = densefuse_fuse(enc,dec,ir,vis)
%DENSEFUSE_FUSE  Fuse an IR/VIS pair: encode both, add features, decode.
ir  = im2single(ir);  if size(ir,3) >1, ir =rgb2gray(ir);  end
vis = im2single(vis); if size(vis,3)>1, vis=rgb2gray(vis); end
if ~isequal(size(ir),size(vis)), vis = imresize(vis,size(ir)); end

XI = dlarray(reshape(ir, size(ir,1), size(ir,2),1,1),'SSCB');
XV = dlarray(reshape(vis,size(vis,1),size(vis,2),1,1),'SSCB');
if canUseGPU, XI = gpuArray(XI); XV = gpuArray(XV); end

fI = forward(enc,XI);
fV = forward(enc,XV);
fused = fI + fV;                 % addition fusion strategy
R = forward(dec,fused);
F = mat2gray(double(gather(extractdata(R))));
end

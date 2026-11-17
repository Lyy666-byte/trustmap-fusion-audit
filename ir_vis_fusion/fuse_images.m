function F = fuse_images(net,ir,vis)
%FUSE_IMAGES  Fuse one IR/VIS pair with a trained network (any image size).
%   ir, vis : grayscale or RGB images (any numeric type).
%   F       : fused grayscale image, double in [0,1].

ir  = im2single(ir);
vis = im2single(vis);
if size(ir,3)  > 1, ir  = rgb2gray(ir);  end
if size(vis,3) > 1, vis = rgb2gray(vis); end

% pad to a common size if needed (use the visible image grid as reference)
if ~isequal(size(ir),size(vis))
    ir = imresize(ir,[size(vis,1) size(vis,2)]);
end

X = cat(3,ir,vis);
X = dlarray(reshape(X,size(X,1),size(X,2),2,1),'SSCB');
if canUseGPU, X = gpuArray(X); end

F = predict(net,X);
F = double(gather(extractdata(F)));
F = mat2gray(F);
end

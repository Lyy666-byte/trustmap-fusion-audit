function count_flops()
%COUNT_FLOPS  FLOPs of the MATLAB networks for one 640x480 image pair.
%   Same convention as measure_sota_flops.py (torch FlopCounterMode): 2 x multiply-accumulates
%   of every convolution (grouped and dilated included), bias and element-wise operations not
%   counted. For a convolution layer, MACs = (output height x width) x numel(weights), which holds
%   for ordinary and grouped weights alike; the output size is read from the network itself.
%   Appends Ours and DenseFuse to results/efficiency_flops_mem.csv.
H = 480; W = 640;
So = load('results/fusion_net_v3.mat');
ours = convFlops(So.net, dlarray(rand(H,W,2,1,'single'),'SSCB'));
D = load('results/densefuse.mat');
X1 = dlarray(rand(H,W,1,1,'single'),'SSCB');
encF = convFlops(D.enc, X1);
Z = forward(D.enc, X1);
decF = convFlops(D.dec, Z);
df = 2*encF + decF;                 % encoder on each source, decoder on the fused feature
fprintf('Ours      GFLOPs = %.3f\n', ours/1e9);
fprintf('DenseFuse GFLOPs = %.3f (encoder %.3f x2, decoder %.3f)\n', df/1e9, encF/1e9, decF/1e9);
writeRow('Ours', ours/1e9); writeRow('DenseFuse', df/1e9);
end

function f = convFlops(net, X)
f = 0;
for i = 1:numel(net.Layers)
    L = net.Layers(i);
    if isa(L,'nnet.cnn.layer.Convolution2DLayer') || isa(L,'nnet.cnn.layer.GroupedConvolution2DLayer')
        Y = predict(net, X, 'Outputs', L.Name);
        sz = size(Y);                                   % S x S x C x B
        f = f + 2 * sz(1) * sz(2) * numel(L.Weights);
        fprintf('  %-14s out %4dx%4dx%3d  weights %6d\n', L.Name, sz(1), sz(2), size(Y,3), numel(L.Weights));
    elseif isa(L,'nnet.cnn.layer.FullyConnectedLayer')
        f = f + 2 * numel(L.Weights);
    end
end
end

function writeRow(name, g)
fn = 'results/efficiency_flops_mem.csv';
T = readtable(fn, 'TextType', 'string', 'Delimiter', ',');
T.method = string(T.method);
T.gflops = string(T.gflops); T.peak_mem_mb = string(T.peak_mem_mb);
k = find(T.method == name);
if isempty(k)
    T = [T; {string(name), string(sprintf('%.4f', g)), ""}];
else
    T.gflops(k) = string(sprintf('%.4f', g));
end
fid = fopen(fn, 'w'); fprintf(fid, 'method,gflops,peak_mem_mb\n');
for i = 1:height(T)
    pm = T.peak_mem_mb(i); if ismissing(pm), pm = ""; end
    fprintf(fid, '%s,%s,%s\n', T.method(i), T.gflops(i), pm);
end
fclose(fid);
end

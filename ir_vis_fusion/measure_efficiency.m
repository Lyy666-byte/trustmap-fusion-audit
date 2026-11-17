function measure_efficiency()
%MEASURE_EFFICIENCY  Inference time and parameter count of the deployed model and
%   of the MATLAB-side DenseFuse baseline, appended to results/efficiency_timing.csv
%   so Table 5 of the paper is generated from a stored artifact rather than from
%   hard-coded numbers. The PyTorch competitors are measured by
%   measure_sota_params.py, which writes to the same file.
So = load('results/fusion_net_v3.mat'); net = So.net;
H = 480; W = 640; N = 50;
X = dlarray(rand(H,W,2,1,'single'),'SSCB');
useGPU = canUseGPU; if useGPU, X = gpuArray(X); end
for i = 1:5, y = predict(net,X); end
t = tic; for i = 1:N, y = predict(net,X); end
if useGPU, wait(gpuDevice); end
ours_ms = toc(t)/N*1000; ours_p = sumLearn(net);
fprintf('ours_infer_ms = %.2f\nours_params = %d\n', ours_ms, ours_p);

D = load('results/densefuse.mat'); enc = D.enc; dec = D.dec;
X1 = dlarray(rand(H,W,1,1,'single'),'SSCB'); if useGPU, X1 = gpuArray(X1); end
for i = 1:5, f1 = forward(enc,X1); r1 = forward(dec,f1); end
t = tic;
for i = 1:N, fi = forward(enc,X1); fv = forward(enc,X1); r1 = forward(dec,fi+fv); end
if useGPU, wait(gpuDevice); end
df_ms = toc(t)/N*1000; df_p = sumLearn(enc)+sumLearn(dec);
fprintf('densefuse_infer_ms = %.2f\ndensefuse_params = %d\n', df_ms, df_p);
fprintf('input_size = %dx%d, GPU = %d\n', H, W, useGPU);

writeRow('Ours',ours_p,ours_ms,'MATLAB');
writeRow('DenseFuse',df_p,df_ms,'MATLAB');
fprintf('appended to results/efficiency_timing.csv\n');
end

function writeRow(name,p,ms,fw)
f = 'results/efficiency_timing.csv';
rows = {}; hdr = 'method,params,time_ms,framework';
if isfile(f)
    T = readtable(f,'TextType','string');
    for i = 1:height(T)
        if ~strcmp(T.method(i),name)
            rows{end+1} = sprintf('%s,%s,%s,%s',T.method(i),string(T.params(i)), ...
                string(T.time_ms(i)),T.framework(i)); %#ok<AGROW>
        end
    end
end
fid = fopen(f,'w'); fprintf(fid,'%s\n',hdr);
for i = 1:numel(rows), fprintf(fid,'%s\n',rows{i}); end
fprintf(fid,'%s,%d,%.2f,%s\n',name,p,ms,fw);
fclose(fid);
end

function n = sumLearn(net)
n = 0; L = net.Learnables;
for i = 1:height(L), n = n + numel(L.Value{i}); end
end

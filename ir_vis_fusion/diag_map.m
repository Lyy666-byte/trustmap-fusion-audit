function T = diag_map(modelPath,outName,nImg,label)
%DIAG_MAP  Quantify the spatial selectivity of an attention/trust map.
%   Reports the within-image dynamic range and, critically, how far the map is
%   from a pure scalar gain (0 = the module is equivalent to a constant).
if nargin < 3 || isempty(nImg), nImg = 30; end
if nargin < 4 || isempty(label), label = outName; end
S = load(modelPath);
if isfield(S,'net'), net = S.net; else, f = fieldnames(S); net = S.(f{1}); end

d = dir(fullfile('tmp_msrs/test/ir','*.png'));
names = sort({d.name}); names = names(1:min(nImg,numel(names)));
N = numel(names);
mMean=zeros(N,1); mStd=zeros(N,1); mCV=zeros(N,1);
mP05=zeros(N,1); mP95=zeros(N,1); mF08=zeros(N,1); mF02=zeros(N,1);
relDev=zeros(N,1); corrIR=zeros(N,1);

for i = 1:N
    nm = names{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end
    M = double(gather(extractdata(predict(net,X,'Outputs',outName))));

    mMean(i)=mean(M(:)); mStd(i)=std(M(:)); mCV(i)=mStd(i)/(mMean(i)+1e-9);
    mP05(i)=prctile(M(:),5); mP95(i)=prctile(M(:),95);
    mF08(i)=mean(M(:)>0.8);  mF02(i)=mean(M(:)<0.2);
    % deviation from replacing the map by its own mean (pure scalar gain)
    relDev(i) = norm(M(:)-mMean(i)) / (norm(M(:))+1e-9);
    % how much the map tracks infrared salient content (local contrast of IR)
    irc = abs(double(ir) - imfilter(double(ir),fspecial('average',15),'replicate'));
    corrIR(i) = corr(M(:),irc(:));
end

fprintf('\n===== %s : %s over %d images =====\n',label,outName,N);
fprintf('mean              : %.4f\n',mean(mMean));
fprintf('within-image std  : %.4f   CV %.4f\n',mean(mStd),mean(mCV));
fprintf('p05 / p95         : %.4f / %.4f   dynamic range %.4f\n',mean(mP05),mean(mP95),mean(mP95-mP05));
fprintf('frac>0.8 / frac<0.2: %.3f / %.3f\n',mean(mF08),mean(mF02));
fprintf('DEVIATION FROM CONSTANT : %.4f\n',mean(relDev));
fprintf('corr with IR local contrast : %.4f\n',mean(corrIR));

T = table(names(:),mMean,mStd,mCV,mP05,mP95,mF08,mF02,relDev,corrIR, ...
 'VariableNames',{'image','mean','std','cv','p05','p95','frac_gt08','frac_lt02','dev_from_const','corr_IRcontrast'});
writetable(T,sprintf('results/diag_%s.csv',label));
end

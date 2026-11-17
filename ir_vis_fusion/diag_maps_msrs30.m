function diag_maps_msrs30()
%DIAG_MAPS_MSRS30  Attention-degeneration statistics for the OLD single-stream
%   CBAM and the NEW cross-modal trust map, computed on EXACTLY the same image
%   set used by every other table in the paper (the MSRS30 subset stored in
%   results/peritem_MSRS.mat), so a single consistent pair of numbers can be
%   quoted throughout the manuscript.
L0 = load('results/peritem_MSRS.mat'); files = L0.files;
So = load('results/fusion_net_v4_OLD_backup.mat'); netOld = So.net;   % single-stream CBAM
Sn = load('results/fusion_net_v3.mat');            netNew = Sn.net;   % deployed V7

N = numel(files);
stat = struct('name',{},'mean',{},'std',{},'cv',{},'p05',{},'p95',{},'dev',{},'f08',{});
maps = {'old','sa_sig',netOld; 'new','trust',netNew};

out = table();
for k = 1:2
    lbl = maps{k,1}; outName = maps{k,2}; net = maps{k,3};
    m=zeros(N,1); s=zeros(N,1); p5=zeros(N,1); p95=zeros(N,1); dv=zeros(N,1); f8=zeros(N,1);
    for i = 1:N
        nm = files{i};
        ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
        vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
        if size(ir,3)>1, ir = rgb2gray(ir); end
        if size(vi,3)>1, vi = rgb2gray(vi); end
        X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
        if canUseGPU, X = gpuArray(X); end
        M = double(gather(extractdata(predict(net,X,'Outputs',outName))));
        m(i)=mean(M(:)); s(i)=std(M(:)); p5(i)=prctile(M(:),5); p95(i)=prctile(M(:),95);
        dv(i)=norm(M(:)-m(i))/(norm(M(:))+1e-9); f8(i)=mean(M(:)>0.8);
    end
    fprintf('\n=== %s (%s) on MSRS30, N=%d ===\n',lbl,outName,N);
    fprintf('  mean              %.4f\n',mean(m));
    fprintf('  within-image std  %.4f\n',mean(s));
    fprintf('  CV                %.4f\n',mean(s./(m+1e-9)));
    fprintf('  p05 / p95         %.4f / %.4f   range %.4f\n',mean(p5),mean(p95),mean(p95-p5));
    fprintf('  frac > 0.8        %.4f\n',mean(f8));
    fprintf('  DEV FROM CONSTANT %.4f\n',mean(dv));
    out = [out; table({lbl},mean(m),mean(s),mean(s./(m+1e-9)),mean(p5),mean(p95),mean(p95-p5),mean(f8),mean(dv), ...
        'VariableNames',{'map','mean','std','cv','p05','p95','range','frac_gt08','dev_from_const'})]; %#ok<AGROW>
end
writetable(out,'results/diag_maps_msrs30.csv');
fprintf('\nwrote results/diag_maps_msrs30.csv\n');
end

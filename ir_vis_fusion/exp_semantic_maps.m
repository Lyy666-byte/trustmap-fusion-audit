function exp_semantic_maps()
%EXP_SEMANTIC_MAPS  Quantitative relation between the spatial maps and labelled objects.
%   On the MSRS30 subset (same 30 test pairs as every table), read the MSRS semantic
%   segmentation labels (tmp_msrs/test/Segmentation_labels; 0 unlabelled, 1 car, 2 person,
%   3 bike, 4 curve, 5 car_stop, 6 guardrail, 7 color_cone, 8 bump) and compare, for
%   pixels of the object classes person/car/bike ("objects") against unlabelled pixels
%   ("background"):
%     - trust map W of the deployed network (layer 'trust')
%     - spatial attention map of the single-stream predecessor (layer 'sa_sig')
%     - two input-only references: infrared intensity, and the saliency prior S
%       (min-max normalised |I_ir - mean_15x15(I_ir)|, as in the loss)
%   Per image: mean inside objects / background, difference, and AUROC of the map for
%   object-vs-background pixels (rank based; 0.5 = no separation).
%   Writes results/semantic_maps_perimage.csv and results/semantic_maps_summary.csv.
L0 = load('results/peritem_MSRS.mat'); files = L0.files;
So = load('results/fusion_net_v4_OLD_backup.mat'); netOld = So.net;
Sn = load('results/fusion_net_v3.mat');            netNew = Sn.net;
N = numel(files);
names = {'W_trust','M_predecessor','IR_intensity','S_saliency'};
K = numel(names);
nObj = zeros(N,1); nBg = zeros(N,1);
muO = nan(N,K); muB = nan(N,K); auc = nan(N,K); muPer = nan(N,K); muCar = nan(N,K);
muBgP = nan(N,K); muBgC = nan(N,K);      % background mean on the images that contain the class
rWir = nan(N,1);
for i = 1:N
    nm = files{i};
    ir = im2single(imread(fullfile('tmp_msrs/test/ir',nm)));
    vi = im2single(imread(fullfile('tmp_msrs/test/vi',nm)));
    if size(ir,3)>1, ir = rgb2gray(ir); end
    if size(vi,3)>1, vi = rgb2gray(vi); end
    lab = imread(fullfile('tmp_msrs/test/Segmentation_labels',nm));
    X = dlarray(reshape(single(cat(3,ir,vi)),[size(ir,1) size(ir,2) 2 1]),'SSCB');
    if canUseGPU, X = gpuArray(X); end
    W = double(gather(extractdata(predict(netNew,X,'Outputs','trust'))));
    M = double(gather(extractdata(predict(netOld,X,'Outputs','sa_sig'))));
    assert(isequal(size(W),size(lab)) && isequal(size(M),size(lab)), 'map size mismatch');
    S = abs(double(ir) - imfilter(double(ir),ones(15)/225,'replicate'));
    S = (S - min(S(:))) / (max(S(:)) - min(S(:)) + eps);
    maps = {W, M, double(ir), S};
    obj = lab==1 | lab==2 | lab==3;  bg = lab==0;
    nObj(i) = nnz(obj); nBg(i) = nnz(bg);
    c = corrcoef(W(:), double(ir(:))); rWir(i) = c(1,2);
    for k = 1:K
        A = maps{k};
        if nObj(i) >= 50
            muO(i,k) = mean(A(obj)); muB(i,k) = mean(A(bg));
            auc(i,k) = auroc(A(obj), A(bg));
        end
        if nnz(lab==2) >= 50, muPer(i,k) = mean(A(lab==2)); muBgP(i,k) = mean(A(bg)); end
        if nnz(lab==1) >= 50, muCar(i,k) = mean(A(lab==1)); muBgC(i,k) = mean(A(bg)); end
    end
    fprintf('%2d %s obj=%6d  AUC W=%.3f M=%.3f IR=%.3f S=%.3f\n',i,nm,nObj(i),auc(i,:));
end
T = table(files(:), nObj, nBg, 'VariableNames', {'file','n_object_px','n_background_px'});
for k = 1:K
    T.(['mean_obj_' names{k}]) = muO(:,k);
    T.(['mean_bg_'  names{k}]) = muB(:,k);
    T.(['auroc_'    names{k}]) = auc(:,k);
end
T.corr_W_IR = rWir;
writetable(T,'results/semantic_maps_perimage.csv');

v = ~isnan(auc(:,1));
fid = fopen('results/semantic_maps_summary.csv','w');
fprintf(fid,'map,n_images,mean_obj,mean_bg,diff,p_diff,auroc_mean,auroc_min,auroc_max,mean_person,mean_car\n');
for k = 1:K
    d = muO(v,k) - muB(v,k);
    fprintf(fid,'%s,%d,%.4f,%.4f,%.4f,%.3g,%.4f,%.4f,%.4f,%.4f,%.4f\n', names{k}, nnz(v), ...
        mean(muO(v,k)), mean(muB(v,k)), mean(d), signrank(d), mean(auc(v,k)), min(auc(v,k)), max(auc(v,k)), ...
        mean(muPer(:,k),'omitnan'), mean(muCar(:,k),'omitnan'));
end
fprintf(fid,'pairs,%d,,,,,,,,,\n', nnz(v));
fprintf(fid,'p_auroc_W_vs_M,%.3g,,,,,,,,,\n', signrank(auc(v,1), auc(v,2)));
fprintf(fid,'p_auroc_W_vs_IR,%.3g,,,,,,,,,\n', signrank(auc(v,1), auc(v,3)));
fprintf(fid,'p_auroc_W_vs_S,%.3g,,,,,,,,,\n', signrank(auc(v,1), auc(v,4)));
fprintf(fid,'mean_corr_W_IR,%.4f,,,,,,,,,\n', mean(rWir));
fprintf(fid,'n_images_with_person,%d,,,,,,,,,\n', nnz(~isnan(muPer(:,1))));
fprintf(fid,'n_images_with_car,%d,,,,,,,,,\n', nnz(~isnan(muCar(:,1))));
fclose(fid);
% per class, paired on the images that contain the class (>= 50 px)
fid = fopen('results/semantic_maps_perclass.csv','w');
fprintf(fid,'map,class,n_images,mean_class,mean_bg_same_images,ratio,p_signrank,n_images_class_gt_bg\n');
for k = 1:K
    for c = 1:2
        if c == 1, a = muPer(:,k); b = muBgP(:,k); cn = 'person';
        else,      a = muCar(:,k); b = muBgC(:,k); cn = 'car'; end
        u = ~isnan(a);
        fprintf(fid,'%s,%s,%d,%.4f,%.4f,%.3f,%.3g,%d\n', names{k}, cn, nnz(u), mean(a(u)), mean(b(u)), ...
            mean(a(u))/mean(b(u)), signrank(a(u),b(u)), nnz(a(u) > b(u)));
    end
end
fclose(fid);
type('results/semantic_maps_perclass.csv');
type('results/semantic_maps_summary.csv');
end

function a = auroc(pos, neg)
% Mann-Whitney AUROC with tie handling: P(pos > neg) + 0.5 P(pos == neg).
x = [pos(:); neg(:)];
r = tiedrank(x);
np = numel(pos); nn = numel(neg);
a = (sum(r(1:np)) - np*(np+1)/2) / (np*nn);
end

function eval_ablation2(nImg)
%EVAL_ABLATION2  Evaluate the four core-design ablation models on MSRS test
%   at full resolution (13 metrics), tying every row to a concrete design
%   choice. RESUMABLE: per-image metrics accumulate in results/abl2_acc.mat so
%   a timeout never leaves a stale/partial table behind. The summary CSV is
%   written ONLY when every image of every model has been computed.
if nargin < 1 || isempty(nImg), nImg = 30; end
% The A3 (full) row is literally the deployed final model (fusion_net_v3.mat),
% which is architecturally identical to build_abl('A3_diff_trust') (verified:
% 37 layers, same names/connections, 43356 params). Evaluated on the SAME
% MSRS30 image list used by the main comparison table, so the ablation ladder
% ends exactly at the reported "Ours".
mods  = {'results/model_AB0.mat','results/model_AB1.mat','results/model_AB2.mat','results/fusion_net_v3.mat'};
labs  = {'A0_single','A1_dual_cat','A2_cat_trust','A3_diff_trust'};
names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};

nets = cell(1,4);
for k=1:4, S=load(mods{k}); f=fieldnames(S); nets{k}=S.(f{strcmp(f,'net')}); end

% same image set as the main table (MSRS30 subset), read from peritem_MSRS.mat
L0 = load('results/peritem_MSRS.mat'); files = L0.files;
if nImg < numel(files), files = files(1:nImg); end
N = numel(files); K = numel(names);

af = 'results/abl2_acc.mat';
if exist(af,'file')
    L=load(af); ACC=L.ACC; DEV=L.DEV;
    if ~isequal(size(ACC),[4 K N]), ACC=nan(4,K,N); DEV=nan(4,N); end
else
    ACC = nan(4,K,N); DEV = nan(4,N);
end

for i=1:N
    if ~any(isnan(ACC(:,:,i)),'all'), fprintf('skip %d/%d\n',i,N); continue; end
    nm=files{i};
    ir=imread(fullfile('tmp_msrs/test/ir',nm)); vi=imread(fullfile('tmp_msrs/test/vi',nm));
    irg=im2double(ir); if size(irg,3)>1, irg=rgb2gray(irg); end
    vig=im2double(vi); if size(vig,3)>1, vig=rgb2gray(vig); end
    for k=1:4
        F = fuse_images(nets{k},ir,vi);
        ACC(k,:,i) = metric_full(F,irg,vig);
        if any(strcmp({nets{k}.Layers.Name},'trust'))
            irs=im2single(irg); vis=im2single(vig);
            X=dlarray(reshape(single(cat(3,irs,vis)),[size(irs,1) size(irs,2) 2 1]),'SSCB');
            if canUseGPU, X=gpuArray(X); end
            W=double(gather(extractdata(predict(nets{k},X,'Outputs','trust'))));
            DEV(k,i)=norm(W(:)-mean(W(:)))/(norm(W(:))+1e-9);
        end
    end
    save(af,'ACC','DEV','files','labs','names');   % incremental (timeout-safe)
    fprintf('done %d/%d %s\n',i,N,nm);
end

if any(isnan(ACC(:)))
    fprintf('PARTIAL: %d images still empty; call again\n',sum(any(any(isnan(ACC),1),2)));
    return
end

MU = mean(ACC,3,'omitnan');
T = array2table(MU,'VariableNames',names,'RowNames',labs);
disp(T);
writetable(T,'results/ablation2_MSRS.csv','WriteRowNames',true);

% paired significance: full model (A3) vs each other config
fid=fopen('results/ablation2_signif.csv','w');
fprintf(fid,'metric,A0,A1,A2,A3,p_A3vsA0,p_A3vsA1,p_A3vsA2\n');
for j=1:K
    v=squeeze(ACC(:,j,:));  % 4 x N
    p=@(a,b) sr(a,b);
    fprintf(fid,'%s,%.4f,%.4f,%.4f,%.4f,%.4g,%.4g,%.4g\n',names{j}, ...
        mean(v(1,:)),mean(v(2,:)),mean(v(3,:)),mean(v(4,:)), ...
        p(v(4,:),v(1,:)),p(v(4,:),v(2,:)),p(v(4,:),v(3,:)));
end
fclose(fid);

fprintf('\nTrust-map deviation from constant (0 = degenerate scalar gain):\n');
for k=1:4, fprintf('  %-14s %.4f\n',labs{k},mean(DEV(k,:),'omitnan')); end
writetable(table(labs(:),mean(DEV,2,'omitnan'),'VariableNames',{'config','trust_dev'}), ...
    'results/ablation2_trustdev.csv');
fprintf('FINALIZED ablation2 (N=%d, 40-epoch matched)\n',N);
end

function p = sr(a,b)
a=a(:); b=b(:); m=~isnan(a)&~isnan(b); p=NaN;
try, if sum(m)>=5, p=signrank(a(m),b(m)); end, catch, end
end

function refresh_mi_abl()
%REFRESH_MI_ABL  Recompute only the MI column of the core-design ablation cache
%   with the corrected estimator, then rewrite the ablation tables.
mods = {'results/model_AB0.mat','results/model_AB1.mat','results/model_AB2.mat','results/fusion_net_v3.mat'};
labs = {'A0_single','A1_dual_cat','A2_cat_trust','A3_diff_trust'};
L = load('results/abl2_acc.mat'); ACC = L.ACC; DEV = L.DEV; files = L.files; names = L.names;
jMI = find(strcmp(names,'MI'));
nets = cell(1,4);
for k=1:4, S=load(mods{k}); f=fieldnames(S); nets{k}=S.(f{strcmp(f,'net')}); end

old = squeeze(ACC(:,jMI,:));
for i = 1:numel(files)
    nm = files{i};
    ir = imread(fullfile('tmp_msrs/test/ir',nm)); vi = imread(fullfile('tmp_msrs/test/vi',nm));
    for k = 1:4
        F = fuse_images(nets{k},ir,vi);
        Fd = im2double(F); ird = im2double(ir); visd = im2double(vi);
        if size(ird,3)>1,  ird  = rgb2gray(ird);  end
        if size(visd,3)>1, visd = rgb2gray(visd); end
        if ~isequal(size(Fd),size(ird)),  ird  = imresize(ird, size(Fd)); end
        if ~isequal(size(Fd),size(visd)), visd = imresize(visd,size(Fd)); end
        ACC(k,jMI,i) = analysis_MI(im2uint8(ird),im2uint8(visd),im2uint8(Fd));
    end
    fprintf('abl MI %d/%d %s\n',i,numel(files),nm);
end
save('results/abl2_acc.mat','ACC','DEV','files','labs','names');
fprintf('abl mean MI before %.3f -> after %.3f\n',mean(old(:),'omitnan'),mean(squeeze(ACC(:,jMI,:)),'all','omitnan'));

MU = mean(ACC,3,'omitnan');
writetable(array2table(MU,'VariableNames',names,'RowNames',labs), ...
    'results/ablation2_MSRS.csv','WriteRowNames',true);
K = numel(names);
fid=fopen('results/ablation2_signif.csv','w');
fprintf(fid,'metric,A0,A1,A2,A3,p_A3vsA0,p_A3vsA1,p_A3vsA2\n');
for j=1:K
    v=squeeze(ACC(:,j,:));
    fprintf(fid,'%s,%.4f,%.4f,%.4f,%.4f,%.4g,%.4g,%.4g\n',names{j}, ...
        mean(v(1,:)),mean(v(2,:)),mean(v(3,:)),mean(v(4,:)), ...
        sr(v(4,:),v(1,:)),sr(v(4,:),v(2,:)),sr(v(4,:),v(3,:)));
end
fclose(fid);
fprintf('ablation tables rewritten\n');
end

function p = sr(a,b)
a=a(:); b=b(:); m=~isnan(a)&~isnan(b); p=NaN;
try, if sum(m)>=5, p=signrank(a(m),b(m)); end, catch, end
end

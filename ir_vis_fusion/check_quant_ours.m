function check_quant_ours()
%CHECK_QUANT_OURS  Section 4 scores our output in floating point, the competitors from their 8-bit PNG files.
%   Score our reported model both ways on the three evaluation subsets, with the Section 4 sources, and compare with
%   DCEvo's 8-bit output. results/check_quant_ours.csv.
S = load(fullfile('results','fusion_net_v3.mat')); net = S.net;
sets = {'MSRS', fullfile('tmp_msrs','test'), 'MSRS_test30';
        'TNO', fullfile('data','TNO'), 'TNO20';
        'RoadScene', fullfile('data','RoadScene'), 'RoadScene40'};
fid = fopen(fullfile('results','check_quant_ours.csv'),'w');
fprintf(fid,'dataset,image,variant,%s\n',strjoin({'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'},','));
for s = 1:size(sets,1)
    T = readtable(fullfile('data_manifest',[sets{s,3} '.csv']),'TextType','string');
    names = unique(T.file(T.modality=="vi"),'stable');
    for i = 1:numel(names)
        nm = char(names(i));
        ir = imread(fullfile(sets{s,2},'ir',nm)); vi = imread(fullfile(sets{s,2},'vi',nm));
        irg = im2double(ir); if size(irg,3)>1, irg = rgb2gray(irg); end
        vig = im2double(vi); if size(vig,3)>1, vig = rgb2gray(vig); end
        F = fuse_images(net,ir,vi);
        d = fullfile('results','sota',['DCEvo_' sets{s,1}]);
        if strcmp(sets{s,1},'MSRS'), d = fullfile('results','sota','DCEvo_MSRS'); end
        V = {metric_full(F,irg,vig), metric_full(im2double(im2uint8(F)),irg,vig), metric_full(imread(fullfile(d,nm)),irg,vig)};
        lab = {'ours_float','ours_uint8','DCEvo_png'};
        for k = 1:3
            fprintf(fid,'%s,%s,%s,%s\n',sets{s,1},nm,lab{k},strjoin(arrayfun(@(x) sprintf('%.6f',x),V{k},'UniformOutput',false),','));
        end
    end
    fprintf('%s %d images\n',sets{s,1},numel(names));
end
fclose(fid);
fprintf('DONE check_quant_ours\n');
end

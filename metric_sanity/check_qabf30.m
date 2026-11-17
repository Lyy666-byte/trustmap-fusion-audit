function check_qabf30()
%CHECK_QABF30  Qabf, SSIM and Nabf of our network and DCEvo through the audit pipeline (8-bit luminance in stim/) on the
%   30 MSRS test images of Section 4 (every twelfth of the 361 in file-name order), to separate the effect of the
%   image sample from that of the pipeline. Exploratory check, added after the audit results; results/check_qabf30.csv.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','ir_vis_fusion'));
S = jsondecode(fileread(fullfile(here,'results','image_sets.json')));
names = S.msrs_all(1:12:end); names = names(1:30);
meths = {'Ours','DCEvo','CDDFuse'};
V = nan(numel(names),numel(meths),3);
b = fullfile(here,'stim','MSRS','base');
parfor i = 1:numel(names)
    ir = im2double(imread(fullfile(b,'ir',names{i}))); vi = im2double(imread(fullfile(b,'vi',names{i})));
    v = nan(numel(meths),3);
    for m = 1:numel(meths)
        F = im2double(imread(fullfile(here,'stim','MSRS',meths{m},'native',names{i})));
        x = metric_full(F,ir,vi);
        v(m,:) = x([9 6 11]);   % Qabf SSIM Nabf
    end
    V(i,:,:) = v;
end
fid = fopen(fullfile(here,'results','check_qabf30.csv'),'w');
fprintf(fid,'image,method,Qabf,SSIM,Nabf\n');
for i = 1:numel(names)
    for m = 1:numel(meths)
        fprintf(fid,'%s,%s,%.6f,%.6f,%.6f\n',names{i},meths{m},V(i,m,1),V(i,m,2),V(i,m,3));
    end
end
fclose(fid);
fprintf('first %s last %s\n',names{1},names{end});
for m = 1:numel(meths)
    fprintf('%-8s Qabf %.4f SSIM %.4f Nabf %.4f\n',meths{m},mean(V(:,m,1)),mean(V(:,m,2)),mean(V(:,m,3)));
end
end

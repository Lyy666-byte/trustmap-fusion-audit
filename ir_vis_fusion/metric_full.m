function [v,names] = metric_full(F,ir,vis)
%METRIC_FULL  Full fusion metric suite.
%   names: EN SD SF AG MI SSIM MS_SSIM CC Qabf FMIw Nabf SCD
%   Higher is better for ALL except Nabf (artifacts -> lower is better).

names = {'EN','SD','SF','AG','MI','SSIM','MS_SSIM','CC','Qabf','FMI','Nabf','SCD','VIF'};

Fd = im2double(F); ird = im2double(ir); visd = im2double(vis);
if size(Fd,3)>1,   Fd   = rgb2gray(Fd);   end
if size(ird,3)>1,  ird  = rgb2gray(ird);  end
if size(visd,3)>1, visd = rgb2gray(visd); end
if ~isequal(size(Fd),size(ird)),  ird  = imresize(ird, size(Fd)); end
if ~isequal(size(Fd),size(visd)), visd = imresize(visd,size(Fd)); end
% Every fused image is scored as the 8-bit luminance image a fusion method delivers (2026-11-17): the released
% methods are read from their 8-bit files, so floating-point outputs (our network, DenseFuse, the traditional
% methods, blends) and colour outputs are rounded to 8-bit luminance here as well.
Fd = im2double(im2uint8(Fd));
Fu = im2uint8(Fd); iru = im2uint8(ird); viu = im2uint8(visd);

EN = entropy(Fd);
SD = std(Fd(:));
rf = sqrt(mean(diff(Fd,1,2).^2,'all'));
cf = sqrt(mean(diff(Fd,1,1).^2,'all'));
SF = sqrt(rf^2 + cf^2);
[gx,gy] = gradient(Fd);
AG = mean(sqrt((gx.^2 + gy.^2)/2),'all');
CC = (corr2(Fd,ird) + corr2(Fd,visd))/2;

MI   = analysis_MI(iru,viu,Fu);
SSIM = (ssim(Fd,ird) + ssim(Fd,visd))/2;
Qabf = analysis_Qabf(ird,visd,Fd);
Nabf = analysis_nabf(Fd,ird,visd);
SCD  = analysis_SCD(ird,visd,Fd);

% FMI / MS-SSIM / VIF are computed at FULL resolution (no downscaling),
% so the values are directly comparable with the literature.
FMI = safe(@() analysis_fmi(ird,visd,Fd));          % default feature, full res
MS_SSIM = safe(@() msssim_wrap(iru,viu,Fu));        % full res
VIF = safe(@() vifp(ird*255,Fd*255) + vifp(visd*255,Fd*255));

v = [EN SD SF AG MI SSIM MS_SSIM CC Qabf FMI Nabf SCD VIF];
end

function ms = msssim_wrap(iru,viu,Fu)
[s1,s2] = size(iru);
imgSeq = zeros(s1,s2,2);
imgSeq(:,:,1) = double(iru);
imgSeq(:,:,2) = double(viu);
ms = analysis_ms_ssim(imgSeq, double(Fu));
end

function y = safe(fn)
try, y = fn(); catch, y = NaN; end
if ~isscalar(y) || ~isreal(y), y = real(y(1)); end
end

function F = classic_fusion(ir,vis,method)
%CLASSIC_FUSION  Classical IR-VIS fusion baselines for comparison.
%   method : 'average' | 'laplacian' | 'dwt' | 'gff'
%   ir,vis : grayscale, any type. F : fused grayscale double in [0,1].

ir = im2double(ir); vis = im2double(vis);
if size(ir,3)  > 1, ir  = rgb2gray(ir);  end
if size(vis,3) > 1, vis = rgb2gray(vis); end
if ~isequal(size(ir),size(vis)), vis = imresize(vis,size(ir)); end

switch lower(method)
    case 'average'
        F = (ir + vis)/2;
    case 'laplacian'
        F = lap_pyr_fusion(ir,vis,4);
    case 'dwt'
        F = dwt_fusion(ir,vis,2);
    case 'gff'
        F = gff_fusion(ir,vis);
    otherwise
        error('Unknown method: %s',method);
end
F = mat2gray(F);
end

% ------------------------------------------------------------------
function F = lap_pyr_fusion(A,B,nlev)
% Laplacian-pyramid fusion: max-abs on detail, average on coarsest band.
gA = cell(nlev,1); gB = cell(nlev,1);
gA{1} = A; gB{1} = B;
for l = 2:nlev
    gA{l} = imresize(imgaussfilt(gA{l-1},1),0.5);
    gB{l} = imresize(imgaussfilt(gB{l-1},1),0.5);
end
lF = cell(nlev,1);
for l = 1:nlev-1
    lA = gA{l} - imresize(gA{l+1},size(gA{l}));
    lB = gB{l} - imresize(gB{l+1},size(gB{l}));
    m  = abs(lA) >= abs(lB);
    lF{l} = lA.*m + lB.*(~m);
end
lF{nlev} = (gA{nlev} + gB{nlev})/2;
F = lF{nlev};
for l = nlev-1:-1:1
    F = imresize(F,size(lF{l})) + lF{l};
end
end

% ------------------------------------------------------------------
function F = dwt_fusion(A,B,nlev)
% Discrete wavelet fusion: average approximation, max-abs detail.
wname = 'db2';
[cA,sA] = wavedec2(A,nlev,wname);
[cB,~ ] = wavedec2(B,nlev,wname);
cF = cB;
nApp = prod(sA(1,:));
cF(1:nApp) = (cA(1:nApp) + cB(1:nApp))/2;        % approximation
idx  = nApp+1:numel(cA);                          % detail coefficients
mask = abs(cA(idx)) >= abs(cB(idx));
cF(idx) = cA(idx).*mask + cB(idx).*(~mask);
F = waverec2(cF,sA,wname);
end

% ------------------------------------------------------------------
function F = gff_fusion(A,B)
% Guided-filter fusion (Li et al., 2013), two-scale, simplified.
avg   = fspecial('average',31);
baseA = imfilter(A,avg,'replicate'); detA = A - baseA;
baseB = imfilter(B,avg,'replicate'); detB = B - baseB;

lapA = abs(imfilter(A,fspecial('laplacian',0),'replicate'));
lapB = abs(imfilter(B,fspecial('laplacian',0),'replicate'));
sA = imfilter(lapA,fspecial('gaussian',[11 11],5),'replicate');
sB = imfilter(lapB,fspecial('gaussian',[11 11],5),'replicate');

PA = double(sA >= sB); PB = 1 - PA;

WbA = imguidedfilter(PA,A,'NeighborhoodSize',[45 45],'DegreeOfSmoothing',0.3);
WbB = imguidedfilter(PB,B,'NeighborhoodSize',[45 45],'DegreeOfSmoothing',0.3);
WdA = imguidedfilter(PA,A,'NeighborhoodSize',[7 7], 'DegreeOfSmoothing',1e-6);
WdB = imguidedfilter(PB,B,'NeighborhoodSize',[7 7], 'DegreeOfSmoothing',1e-6);

sb = WbA + WbB + eps; WbA = WbA./sb; WbB = WbB./sb;
sd = WdA + WdB + eps; WdA = WdA./sd; WdB = WdB./sd;

F = WbA.*baseA + WbB.*baseB + WdA.*detA + WdB.*detB;
end

function allX = prepare_data(irDir,visDir,patchSize,stride,maxPairs)
%PREPARE_DATA  Build training tensor from a folder of IR/VIS image pairs.
%   irDir, visDir : folders with matching file names (same name = a pair).
%   patchSize     : patch side length (default 128)
%   stride        : sliding-window stride (default 96)
%   maxPairs      : cap on number of image pairs to read (default Inf)
%
%   Returns allX : patchSize x patchSize x 2 x N (single, [0,1]),
%                  channel 1 = IR, channel 2 = VIS.
%
%   Pairs are matched by file name. IR images are converted to grayscale.
%   VIS images are converted to grayscale (luminance) for the fusion model;
%   color can be re-injected later via YCbCr at inference time.

if nargin < 3 || isempty(patchSize), patchSize = 128; end
if nargin < 4 || isempty(stride),    stride    = 96;  end
if nargin < 5 || isempty(maxPairs),  maxPairs  = Inf; end

irFiles = dir(fullfile(irDir,'*.*'));
irFiles = irFiles(~[irFiles.isdir]);

patches = {};   % collect, concatenate once at the end
np = 0;
for i = 1:numel(irFiles)
    if np >= maxPairs, break; end
    name   = irFiles(i).name;
    irPath = fullfile(irDir,name);
    vPath  = fullfile(visDir,name);
    if ~isfile(vPath), continue; end           % no matching VIS -> skip

    ir  = readGray(irPath);
    vis = readGray(vPath);
    if ~isequal(size(ir),size(vis))
        vis = imresize(vis,size(ir));
    end
    np = np + 1;

    [H,W] = size(ir);
    for r = 1:stride:max(H-patchSize+1,1)
        for c = 1:stride:max(W-patchSize+1,1)
            rr = r:min(r+patchSize-1,H);
            cc = c:min(c+patchSize-1,W);
            if numel(rr) < patchSize || numel(cc) < patchSize, continue; end
            blk = cat(3, ir(rr,cc), vis(rr,cc));
            patches{end+1} = single(blk); %#ok<AGROW>
        end
    end
end

if isempty(patches)
    error('prepare_data:noPairs', ...
        'No matching IR/VIS pairs found. Check folder names match.');
end

allX = cat(4, patches{:});
fprintf('prepare_data: %d pairs -> %d patches of %dx%d\n', ...
        np, size(allX,4), patchSize, patchSize);
end

function g = readGray(p)
g = imread(p);
if size(g,3) > 1, g = rgb2gray(g); end
g = im2single(g);
end

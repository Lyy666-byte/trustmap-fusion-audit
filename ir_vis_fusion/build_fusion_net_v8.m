function net = build_fusion_net_v8(C)
%BUILD_FUSION_NET_V8  V7 with scale-anchored modality features (identifiable trust map).
%
%   Identical to build_fusion_net_v7 except that, before the difference D and the convex fusion,
%   each modality feature is divided by its own root-mean-square over the image and channels
%   (per sample, no learnable parameters). Rescaling an encoder branch then no longer changes the
%   scale of F_ir or F_vis, so it can no longer absorb a change of the level of W (Section 3.3).
%
%   (V7 description follows.)
%
%   V7 = V6 (modality-specific grouped encoder, cross-modal differential trust
%   map, explicit convex modality selection) plus two fixes found empirically:
%     * a post-fusion refinement residual block, because the convex combination
%       F = F_vis + W.*D constrains the fused feature to the segment between the
%       two modality features and by itself loses representational capacity;
%     * a dilated conv in the trust head so W is decided from regional context
%       rather than a 5x5 neighbourhood.
%   Parameter budget is matched to the single-stream CBAM baseline (~4.4e4) so
%   that any quality difference is attributable to the architecture, not size.
%
%   Named outputs for visualisation: 'trust' (W), 'diffm' (D), 'fused'.

if nargin < 1 || isempty(C), C = 20; end
C2 = 2*C;
Cr = max(4, round(C2/4));

lg = layerGraph();

% ---- modality-specific stem (grouped: 2 groups = {IR, VIS}) -------------
lg = addLayers(lg,[
    groupedConvolution2dLayer(3,C,2,'Padding','same','Name','stem')
    reluLayer('Name','stem_r')]);

% ---- modality-specific residual block 1 ---------------------------------
lg = addLayers(lg,[
    groupedConvolution2dLayer(3,C,2,'Padding','same','Name','rb1_c1')
    reluLayer('Name','rb1_r1')
    groupedConvolution2dLayer(3,C,2,'Padding','same','Name','rb1_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rb1_add'));
lg = addLayers(lg, reluLayer('Name','rb1_o'));
lg = connectLayers(lg,'stem_r','rb1_c1');
lg = connectLayers(lg,'stem_r','rb1_add/in1');
lg = connectLayers(lg,'rb1_c2','rb1_add/in2');
lg = connectLayers(lg,'rb1_add','rb1_o');

% ---- modality-specific residual block 2 ---------------------------------
lg = addLayers(lg,[
    groupedConvolution2dLayer(3,C,2,'Padding','same','Name','rb2_c1')
    reluLayer('Name','rb2_r1')
    groupedConvolution2dLayer(3,C,2,'Padding','same','Name','rb2_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rb2_add'));
lg = addLayers(lg, reluLayer('Name','feat'));
lg = connectLayers(lg,'rb1_o','rb2_c1');
lg = connectLayers(lg,'rb1_o','rb2_add/in1');
lg = connectLayers(lg,'rb2_c2','rb2_add/in2');
lg = connectLayers(lg,'rb2_add','feat');

% ---- cross-modal channel attention --------------------------------------
lg = addLayers(lg,[
    globalAveragePooling2dLayer('Name','ca_gap')
    convolution2dLayer(1,Cr,'Name','ca_fc1')
    reluLayer('Name','ca_relu')
    convolution2dLayer(1,C2,'Name','ca_fc2')
    sigmoidLayer('Name','ca_sig')]);
lg = addLayers(lg, multiplicationLayer(2,'Name','featq'));
lg = connectLayers(lg,'feat','ca_gap');
lg = connectLayers(lg,'feat','featq/in1');
lg = connectLayers(lg,'ca_sig','featq/in2');

% ---- cross-modal differential features and visible base -----------------
lg = addLayers(lg, functionLayer(@(X) cat(3, ...
        X(:,:,1:C,:)       ./ (sqrt(mean(X(:,:,1:C,:).^2,[1 2 3]))       + 1e-6), ...
        X(:,:,C+1:2*C,:)   ./ (sqrt(mean(X(:,:,C+1:2*C,:).^2,[1 2 3])) + 1e-6)), ...
    'Name','fnorm','Formattable',false,'Acceleratable',true));
lg = connectLayers(lg,'featq','fnorm');
lg = addLayers(lg, functionLayer(@(X) X(:,:,1:C,:) - X(:,:,C+1:2*C,:), ...
    'Name','diffm','Formattable',false,'Acceleratable',true));
lg = connectLayers(lg,'fnorm','diffm');
lg = addLayers(lg, functionLayer(@(X) X(:,:,C+1:2*C,:), ...
    'Name','fvis','Formattable',false,'Acceleratable',true));
lg = connectLayers(lg,'fnorm','fvis');

% ---- trust map W (regional context via a dilated conv) ------------------
lg = addLayers(lg,[
    convolution2dLayer(3,8,'Padding','same','Name','w_c1')
    reluLayer('Name','w_r1')
    convolution2dLayer(3,8,'Padding','same','DilationFactor',2,'Name','w_c2')
    reluLayer('Name','w_r2')
    convolution2dLayer(3,1,'Padding','same','Name','w_c3')
    sigmoidLayer('Name','trust')]);
lg = connectLayers(lg,'diffm','w_c1');

% ---- explicit soft modality selection: F = F_vis + W .* D ---------------
lg = addLayers(lg, multiplicationLayer(2,'Name','wd'));
lg = connectLayers(lg,'trust','wd/in1');
lg = connectLayers(lg,'diffm','wd/in2');
lg = addLayers(lg, additionLayer(2,'Name','fusef'));
lg = connectLayers(lg,'fvis','fusef/in1');
lg = connectLayers(lg,'wd','fusef/in2');

% ---- post-fusion refinement residual block ------------------------------
lg = addLayers(lg,[
    convolution2dLayer(3,C,'Padding','same','Name','rf_c1')
    reluLayer('Name','rf_r1')
    convolution2dLayer(3,C,'Padding','same','Name','rf_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rf_add'));
lg = addLayers(lg, reluLayer('Name','rf_o'));
lg = connectLayers(lg,'fusef','rf_c1');
lg = connectLayers(lg,'fusef','rf_add/in1');
lg = connectLayers(lg,'rf_c2','rf_add/in2');
lg = connectLayers(lg,'rf_add','rf_o');

% ---- decoder ------------------------------------------------------------
lg = addLayers(lg,[
    convolution2dLayer(3,C,'Padding','same','Name','dec1')
    reluLayer('Name','dec_r')
    convolution2dLayer(3,1,'Padding','same','Name','dec2')
    sigmoidLayer('Name','fused')]);
lg = connectLayers(lg,'rf_o','dec1');

net = dlnetwork(lg,'Initialize',false);
net = initialize(net, dlarray(zeros(64,64,2,1,'single'),'SSCB'));
end

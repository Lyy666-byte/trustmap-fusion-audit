function net = build_abl(mode,C)
%BUILD_ABL  Ablation builders that isolate the THREE core design choices,
%   so the ablation is tied to the paper's contributions (reviewer comment 6).
%
%   mode = 'A0_single'  : single-stream encoder (features from concatenated
%                         [IR,VIS]), no branch separation, no trust map.
%                         Fusion = decoder(features).  (≈ old design family)
%   mode = 'A1_dual_cat': dual-branch grouped encoder, but fuse the two streams
%                         by plain concatenation + 1x1 conv (NO trust map).
%   mode = 'A2_cat_trust': dual-branch + trust map derived from the
%                         CONCATENATION [F_ir,F_vis] (not the difference).
%   mode = 'A3_diff_trust': dual-branch + trust map from the DIFFERENCE
%                         D=F_ir-F_vis, explicit convex selection == full model.
%
%   All modes share the same depth (stem + 2 res blocks + channel attention +
%   refinement + decoder) and are matched in width, so differences isolate the
%   mechanism, not capacity.

if nargin < 2 || isempty(C), C = 20; end
if nargin < 1 || isempty(mode), mode = 'A3_diff_trust'; end
C2 = 2*C; Cr = max(4,round(C2/4));
lg = layerGraph();

single = strcmp(mode,'A0_single');
grp = 2; if single, grp = 1; end        % single-stream shares weights (1 group)

% ---- stem ----
if single
    lg = addLayers(lg,[convolution2dLayer(3,C2,'Padding','same','Name','stem')
                       reluLayer('Name','stem_r')]);
else
    lg = addLayers(lg,[groupedConvolution2dLayer(3,C,grp,'Padding','same','Name','stem')
                       reluLayer('Name','stem_r')]);
end

% ---- 2 residual blocks (grouped unless single-stream) ----
prev = 'stem_r';
for r = 1:2
    p = sprintf('rb%d',r);
    if single
        lg = addLayers(lg,[convolution2dLayer(3,C2,'Padding','same','Name',[p '_c1'])
                           reluLayer('Name',[p '_r1'])
                           convolution2dLayer(3,C2,'Padding','same','Name',[p '_c2'])]);
    else
        lg = addLayers(lg,[groupedConvolution2dLayer(3,C,grp,'Padding','same','Name',[p '_c1'])
                           reluLayer('Name',[p '_r1'])
                           groupedConvolution2dLayer(3,C,grp,'Padding','same','Name',[p '_c2'])]);
    end
    lg = addLayers(lg, additionLayer(2,'Name',[p '_add']));
    oname = [p '_o']; if r==2, oname='feat'; end
    lg = addLayers(lg, reluLayer('Name',oname));
    lg = connectLayers(lg,prev,[p '_c1']);
    lg = connectLayers(lg,prev,[p '_add/in1']);
    lg = connectLayers(lg,[p '_c2'],[p '_add/in2']);
    lg = connectLayers(lg,[p '_add'],oname);
    prev = oname;
end

% ---- channel attention (on the 2C feature tensor) ----
lg = addLayers(lg,[globalAveragePooling2dLayer('Name','ca_gap')
                   convolution2dLayer(1,Cr,'Name','ca_fc1')
                   reluLayer('Name','ca_relu')
                   convolution2dLayer(1,C2,'Name','ca_fc2')
                   sigmoidLayer('Name','ca_sig')]);
lg = addLayers(lg, multiplicationLayer(2,'Name','featq'));
lg = connectLayers(lg,'feat','ca_gap');
lg = connectLayers(lg,'feat','featq/in1');
lg = connectLayers(lg,'ca_sig','featq/in2');

% ---- fusion mechanism ----
switch mode
    case 'A0_single'
        % features already mixed; go straight to refinement
        fuseOut = 'featq_proj';
        lg = addLayers(lg, convolution2dLayer(1,C,'Padding','same','Name',fuseOut));
        lg = connectLayers(lg,'featq',fuseOut);

    case 'A1_dual_cat'
        % plain learned mix of the two streams, no trust map
        fuseOut = 'catmix';
        lg = addLayers(lg, convolution2dLayer(1,C,'Padding','same','Name',fuseOut));
        lg = connectLayers(lg,'featq',fuseOut);

    case 'A2_cat_trust'
        % trust map from the CONCATENATION of both streams
        lg = addLayers(lg, functionLayer(@(X) X(:,:,1:C,:)-X(:,:,C+1:2*C,:), ...
            'Name','diffm','Formattable',false,'Acceleratable',true));
        lg = connectLayers(lg,'featq','diffm');
        lg = addLayers(lg, functionLayer(@(X) X(:,:,C+1:2*C,:), ...
            'Name','fvis','Formattable',false,'Acceleratable',true));
        lg = connectLayers(lg,'featq','fvis');
        lg = addLayers(lg,[convolution2dLayer(3,8,'Padding','same','Name','w_c1')
                           reluLayer('Name','w_r1')
                           convolution2dLayer(3,8,'Padding','same','DilationFactor',2,'Name','w_c2')
                           reluLayer('Name','w_r2')
                           convolution2dLayer(3,1,'Padding','same','Name','w_c3')
                           sigmoidLayer('Name','trust')]);
        lg = connectLayers(lg,'featq','w_c1');        % <-- driven by concat
        lg = addLayers(lg, multiplicationLayer(2,'Name','wd'));
        lg = connectLayers(lg,'trust','wd/in1');
        lg = connectLayers(lg,'diffm','wd/in2');
        lg = addLayers(lg, additionLayer(2,'Name','fusef'));
        lg = connectLayers(lg,'fvis','fusef/in1');
        lg = connectLayers(lg,'wd','fusef/in2');
        fuseOut = 'fusef';

    case 'A3_diff_trust'
        lg = addLayers(lg, functionLayer(@(X) X(:,:,1:C,:)-X(:,:,C+1:2*C,:), ...
            'Name','diffm','Formattable',false,'Acceleratable',true));
        lg = connectLayers(lg,'featq','diffm');
        lg = addLayers(lg, functionLayer(@(X) X(:,:,C+1:2*C,:), ...
            'Name','fvis','Formattable',false,'Acceleratable',true));
        lg = connectLayers(lg,'featq','fvis');
        lg = addLayers(lg,[convolution2dLayer(3,8,'Padding','same','Name','w_c1')
                           reluLayer('Name','w_r1')
                           convolution2dLayer(3,8,'Padding','same','DilationFactor',2,'Name','w_c2')
                           reluLayer('Name','w_r2')
                           convolution2dLayer(3,1,'Padding','same','Name','w_c3')
                           sigmoidLayer('Name','trust')]);
        lg = connectLayers(lg,'diffm','w_c1');        % <-- driven by difference
        lg = addLayers(lg, multiplicationLayer(2,'Name','wd'));
        lg = connectLayers(lg,'trust','wd/in1');
        lg = connectLayers(lg,'diffm','wd/in2');
        lg = addLayers(lg, additionLayer(2,'Name','fusef'));
        lg = connectLayers(lg,'fvis','fusef/in1');
        lg = connectLayers(lg,'wd','fusef/in2');
        fuseOut = 'fusef';
    otherwise
        error('unknown mode %s',mode);
end

% ---- refinement residual block ----
lg = addLayers(lg,[convolution2dLayer(3,C,'Padding','same','Name','rf_c1')
                   reluLayer('Name','rf_r1')
                   convolution2dLayer(3,C,'Padding','same','Name','rf_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rf_add'));
lg = addLayers(lg, reluLayer('Name','rf_o'));
lg = connectLayers(lg,fuseOut,'rf_c1');
lg = connectLayers(lg,fuseOut,'rf_add/in1');
lg = connectLayers(lg,'rf_c2','rf_add/in2');
lg = connectLayers(lg,'rf_add','rf_o');

% ---- decoder ----
lg = addLayers(lg,[convolution2dLayer(3,C,'Padding','same','Name','dec1')
                   reluLayer('Name','dec_r')
                   convolution2dLayer(3,1,'Padding','same','Name','dec2')
                   sigmoidLayer('Name','fused')]);
lg = connectLayers(lg,'rf_o','dec1');

net = dlnetwork(lg,'Initialize',false);
net = initialize(net, dlarray(zeros(64,64,2,1,'single'),'SSCB'));
end

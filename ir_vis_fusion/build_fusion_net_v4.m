function net = build_fusion_net_v4()
%BUILD_FUSION_NET_V4  Residual + attention lightweight fusion network.
%   Encoder: Conv(2->32) + 2 residual blocks(32) ; CBAM attention ; decoder.
%   Fully convolutional (no input layer) -> any size. Input 2ch [IR,VIS].
lg = layerGraph();

lg = addLayers(lg,[
    convolution2dLayer(3,32,'Padding','same','Name','in')
    reluLayer('Name','in_r')]);

% ---- residual block 1 ----
lg = addLayers(lg,[
    convolution2dLayer(3,32,'Padding','same','Name','rb1_c1')
    reluLayer('Name','rb1_r1')
    convolution2dLayer(3,32,'Padding','same','Name','rb1_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rb1_add'));
lg = addLayers(lg, reluLayer('Name','rb1_o'));
lg = connectLayers(lg,'in_r','rb1_c1');
lg = connectLayers(lg,'in_r','rb1_add/in1');
lg = connectLayers(lg,'rb1_c2','rb1_add/in2');
lg = connectLayers(lg,'rb1_add','rb1_o');

% ---- residual block 2 ----
lg = addLayers(lg,[
    convolution2dLayer(3,32,'Padding','same','Name','rb2_c1')
    reluLayer('Name','rb2_r1')
    convolution2dLayer(3,32,'Padding','same','Name','rb2_c2')]);
lg = addLayers(lg, additionLayer(2,'Name','rb2_add'));
lg = addLayers(lg, reluLayer('Name','feat'));
lg = connectLayers(lg,'rb1_o','rb2_c1');
lg = connectLayers(lg,'rb1_o','rb2_add/in1');
lg = connectLayers(lg,'rb2_c2','rb2_add/in2');
lg = connectLayers(lg,'rb2_add','feat');

% ---- channel attention ----
lg = addLayers(lg,[
    globalAveragePooling2dLayer('Name','ca_gap')
    convolution2dLayer(1,8,'Name','ca_fc1')
    reluLayer('Name','ca_relu')
    convolution2dLayer(1,32,'Name','ca_fc2')
    sigmoidLayer('Name','ca_sig')]);
lg = addLayers(lg, multiplicationLayer(2,'Name','ca_mul'));
lg = connectLayers(lg,'feat','ca_gap');
lg = connectLayers(lg,'feat','ca_mul/in1');
lg = connectLayers(lg,'ca_sig','ca_mul/in2');

% ---- spatial attention ----
lg = addLayers(lg,[
    convolution2dLayer(7,1,'Padding','same','Name','sa_conv')
    sigmoidLayer('Name','sa_sig')]);
lg = addLayers(lg, multiplicationLayer(2,'Name','sa_mul'));
lg = connectLayers(lg,'ca_mul','sa_conv');
lg = connectLayers(lg,'ca_mul','sa_mul/in1');
lg = connectLayers(lg,'sa_sig','sa_mul/in2');

% ---- decoder ----
lg = addLayers(lg,[
    convolution2dLayer(3,16,'Padding','same','Name','dec1')
    reluLayer('Name','dec_r')
    convolution2dLayer(3,1,'Padding','same','Name','dec2')
    sigmoidLayer('Name','fused')]);
lg = connectLayers(lg,'sa_mul','dec1');

net = dlnetwork(lg,'Initialize',false);
net = initialize(net, dlarray(zeros(64,64,2,1,'single'),'SSCB'));
end

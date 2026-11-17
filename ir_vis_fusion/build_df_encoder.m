function net = build_df_encoder()
%BUILD_DF_ENCODER  DenseFuse encoder (Li & Wu, 2019): C1 + dense block -> 64ch.
%   Fully convolutional, 1-channel input, 64-channel feature output.
lg = layerGraph();
lg = addLayers(lg, convolution2dLayer(3,16,'Padding','same','Name','c1'));
lg = addLayers(lg, reluLayer('Name','r1'));
lg = connectLayers(lg,'c1','r1');

lg = addLayers(lg, [convolution2dLayer(3,16,'Padding','same','Name','dc1'); reluLayer('Name','dr1')]);
lg = connectLayers(lg,'r1','dc1');
lg = addLayers(lg, concatenationLayer(3,2,'Name','cat1'));   % 16+16 = 32
lg = connectLayers(lg,'r1','cat1/in1');
lg = connectLayers(lg,'dr1','cat1/in2');

lg = addLayers(lg, [convolution2dLayer(3,16,'Padding','same','Name','dc2'); reluLayer('Name','dr2')]);
lg = connectLayers(lg,'cat1','dc2');
lg = addLayers(lg, concatenationLayer(3,2,'Name','cat2'));   % 32+16 = 48
lg = connectLayers(lg,'cat1','cat2/in1');
lg = connectLayers(lg,'dr2','cat2/in2');

lg = addLayers(lg, [convolution2dLayer(3,16,'Padding','same','Name','dc3'); reluLayer('Name','dr3')]);
lg = connectLayers(lg,'cat2','dc3');
lg = addLayers(lg, concatenationLayer(3,2,'Name','feat'));   % 48+16 = 64
lg = connectLayers(lg,'cat2','feat/in1');
lg = connectLayers(lg,'dr3','feat/in2');

net = dlnetwork(lg,'Initialize',false);
net = initialize(net, dlarray(zeros(64,64,1,1,'single'),'SSCB'));
end

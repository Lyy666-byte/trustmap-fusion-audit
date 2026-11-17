function net = build_df_decoder()
%BUILD_DF_DECODER  DenseFuse decoder: 64 -> 64 -> 32 -> 16 -> 1 (sigmoid).
layers = [
    convolution2dLayer(3,64,'Padding','same','Name','d1')
    reluLayer('Name','dr1')
    convolution2dLayer(3,32,'Padding','same','Name','d2')
    reluLayer('Name','dr2')
    convolution2dLayer(3,16,'Padding','same','Name','d3')
    reluLayer('Name','dr3')
    convolution2dLayer(3,1, 'Padding','same','Name','d4')
    sigmoidLayer('Name','rec')];
net = dlnetwork(layers,'Initialize',false);
net = initialize(net, dlarray(zeros(64,64,64,1,'single'),'SSCB'));
end

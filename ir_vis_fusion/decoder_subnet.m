function dnet = decoder_subnet(net)
%DECODER_SUBNET  Rebuild rf_c1..fused as a standalone network with the trained
%   weights, so the fused feature can be supplied directly.
names = {'rf_c1','rf_r1','rf_c2','rf_add','rf_o','dec1','dec_r','dec2','fused'};
lg = layerGraph();
for k = 1:numel(names)
    l = net.Layers(strcmp({net.Layers.Name},names{k}));
    lg = addLayers(lg,l);
end
lg = connectLayers(lg,'rf_c1','rf_r1');
lg = connectLayers(lg,'rf_r1','rf_c2');
lg = connectLayers(lg,'rf_c2','rf_add/in2');
lg = connectLayers(lg,'rf_add','rf_o');
lg = connectLayers(lg,'rf_o','dec1');
lg = connectLayers(lg,'dec1','dec_r');
lg = connectLayers(lg,'dec_r','dec2');
lg = connectLayers(lg,'dec2','fused');
% the residual skip enters rf_add/in1 straight from the input; a pass-through
% functionLayer is the entry point so the net stays fully convolutional
% (no imageInputLayer => arbitrary image size, as in build_fusion_net_v7.m)
inp = functionLayer(@(x) x,'Name','fin','Formattable',false,'Acceleratable',true);
lg = addLayers(lg,inp);
lg = connectLayers(lg,'fin','rf_c1');
lg = connectLayers(lg,'fin','rf_add/in1');
dnet = dlnetwork(lg,'Initialize',false);
dnet = initialize(dnet, dlarray(zeros(32,32,20,1,'single'),'SSCB'));
% copy trained learnables by (layer,parameter) name
Ls = net.Learnables; Ld = dnet.Learnables;
for i = 1:height(Ld)
    m = strcmp(string(Ls.Layer),string(Ld.Layer(i))) & ...
        strcmp(string(Ls.Parameter),string(Ld.Parameter(i)));
    if any(m), Ld.Value{i} = Ls.Value{find(m,1)}; end
end
dnet.Learnables = Ld;
end


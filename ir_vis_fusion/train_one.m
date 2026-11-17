function train_one(tag,builderName,lossName,epochs,nPairs)
%TRAIN_ONE  Train one ablation/model config on a fixed MSRS subset.
%   Same data + same setup for every config => controlled ablation.
if nargin < 4 || isempty(epochs), epochs = 50;  end
if nargin < 5 || isempty(nPairs), nPairs = 200; end
rng(0);
allX = prepare_data('tmp_msrs/train/ir','tmp_msrs/train/vi',128,128,nPairs);
net  = feval(builderName);
net  = train_fusion(net, allX, epochs, 16, 1e-3, str2func(lossName));
if ~exist('results','dir'), mkdir results; end
save(['results/abl_' tag '.mat'],'net');
if strcmp(tag,'M3') || strcmp(tag,'M4')
    save('results/fusion_net_v3.mat','net');   % final model reused by comparison
end
fprintf('saved results/abl_%s.mat\n',tag);
end

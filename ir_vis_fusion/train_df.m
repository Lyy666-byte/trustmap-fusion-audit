%TRAIN_DF  Train DenseFuse autoencoder on MSRS patches (IR+VIS pooled).
clear; clc; rng(0);

allX = prepare_data('tmp_msrs/train/ir','tmp_msrs/train/vi',128,128,150);
allX1 = cat(4, allX(:,:,1,:), allX(:,:,2,:));   % H x W x 1 x 2N (single channel)
fprintf('DenseFuse training set: %d single-channel patches\n',size(allX1,4));

enc = build_df_encoder();
dec = build_df_decoder();
[enc,dec] = train_densefuse(enc,dec,allX1,30,16,1e-3);
save('results/densefuse.mat','enc','dec');

% sanity check
ir = imread('data/TNO/ir/1.png'); vi = imread('data/TNO/vi/1.png');
F  = densefuse_fuse(enc,dec,ir,vi);
imwrite(F,'results/densefuse_sample.png');
fprintf('DenseFuse trained, saved results/densefuse.mat\n');

function v = vifp(ref,dist)
%VIFP  Pixel-domain Visual Information Fidelity (Sheikh & Bovik).
%   ref, dist : grayscale images, double in [0,255]. Higher is better.
sigma_nsq = 2;
num = 0; den = 0;
ref = double(ref); dist = double(dist);
for scale = 1:4
    N = 2^(4-scale+1) + 1;
    win = fspecial('gaussian', N, N/5);
    if scale > 1
        ref  = filter2(win,ref,'valid');  ref  = ref(1:2:end,1:2:end);
        dist = filter2(win,dist,'valid'); dist = dist(1:2:end,1:2:end);
    end
    mu1 = filter2(win,ref,'valid');  mu2 = filter2(win,dist,'valid');
    mu1_sq = mu1.^2; mu2_sq = mu2.^2; mu1_mu2 = mu1.*mu2;
    s1 = filter2(win,ref.^2, 'valid') - mu1_sq;
    s2 = filter2(win,dist.^2,'valid') - mu2_sq;
    s12 = filter2(win,ref.*dist,'valid') - mu1_mu2;
    s1(s1<0) = 0; s2(s2<0) = 0;
    g = s12 ./ (s1 + 1e-10);
    sv = s2 - g.*s12;
    g(s1<1e-10) = 0; sv(s1<1e-10) = s2(s1<1e-10); s1(s1<1e-10) = 0;
    g(s2<1e-10) = 0; sv(s2<1e-10) = 0;
    sv(g<0) = s2(g<0); g(g<0) = 0;
    sv(sv<=1e-10) = 1e-10;
    num = num + sum(sum(log10(1 + g.^2 .* s1 ./ (sv + sigma_nsq))));
    den = den + sum(sum(log10(1 + s1 ./ sigma_nsq)));
end
v = num / (den + 1e-12);
end
